"""OpenMM implicit-solvent (GBn2) minimization + MD driver.

Honesty rule (brief 0.2): the actual simulated length in ns is measured from
wall-clock throughput on THIS machine (a short speed probe), never assumed
from config — the config value is a *target*, the achieved value is what
gets logged and reported.
"""
from __future__ import annotations

import time
from pathlib import Path

from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform
from openmm.app import (CutoffNonPeriodic, ForceField, HBonds, Modeller, NoCutoff, PDBFile,
                          Simulation, StateDataReporter)
from openmm.unit import femtoseconds, kelvin, kilojoule_per_mole, nanometer, picosecond, picoseconds

from sentinel.utils.logging import get_logger

logger = get_logger(__name__)

# All-pairs (NoCutoff) GBSA is O(N^2) and becomes impractically slow on a
# 2-core CPU above a few hundred atoms (measured directly on this machine:
# the 3420-atom fibril-tip system did not finish 100 minimization iterations
# in >4 minutes with NoCutoff — see PROGRESS_LOG.md M4). Above this atom
# count we switch to a cutoff nonbonded scheme, a standard, documented
# approximation for implicit-solvent GBSA on larger systems.
CUTOFF_ATOM_THRESHOLD = 800
NONBONDED_CUTOFF_NM = 1.5


def build_simulation(pdb_path: str, forcefield_files: list[str], temperature_K: float,
                       friction_per_ps: float, timestep_fs: float):
    pdb = PDBFile(str(pdb_path))
    forcefield = ForceField(*forcefield_files)
    modeller = Modeller(pdb.topology, pdb.positions)
    n_atoms = modeller.topology.getNumAtoms()
    if n_atoms > CUTOFF_ATOM_THRESHOLD:
        system = forcefield.createSystem(modeller.topology, nonbondedMethod=CutoffNonPeriodic,
                                           nonbondedCutoff=NONBONDED_CUTOFF_NM * nanometer,
                                           constraints=HBonds)
        logger.info(f"{n_atoms} atoms > {CUTOFF_ATOM_THRESHOLD}: using CutoffNonPeriodic "
                    f"({NONBONDED_CUTOFF_NM} nm) instead of all-pairs NoCutoff.")
    else:
        system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff,
                                           constraints=HBonds)
    integrator = LangevinMiddleIntegrator(temperature_K * kelvin, friction_per_ps / picosecond,
                                            timestep_fs * femtoseconds)
    platform = Platform.getPlatformByName("CPU")
    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    return simulation


def minimize(simulation, max_iterations: int) -> dict:
    state_before = simulation.context.getState(getEnergy=True)
    simulation.minimizeEnergy(maxIterations=max_iterations)
    state_after = simulation.context.getState(getEnergy=True)
    return {
        "energy_before_kJ_mol": state_before.getPotentialEnergy().value_in_unit(kilojoule_per_mole),
        "energy_after_kJ_mol": state_after.getPotentialEnergy().value_in_unit(kilojoule_per_mole),
    }


def probe_steps_per_second(simulation, n_probe_steps: int = 200) -> float:
    start = time.time()
    simulation.step(n_probe_steps)
    elapsed = time.time() - start
    return n_probe_steps / elapsed if elapsed > 0 else 0.0


def run_md(pdb_path: str, forcefield_files: list[str], temperature_K: float, friction_per_ps: float,
           timestep_fs: float, minimize_max_iterations: int, target_ns: float,
           report_interval_steps: int, out_dir: Path, tag: str, seed: int,
           max_wallclock_s: float = 240.0) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    simulation = build_simulation(pdb_path, forcefield_files, temperature_K, friction_per_ps, timestep_fs)

    # A real bug was found and fixed here: velocities were previously assigned BEFORE
    # minimization. Minimization can move a badly-clashed starting structure (e.g. from
    # PDBFixer-rebuilt side chains on imperfect backbone geometry) a long way to reach a
    # relaxed, low-energy configuration -- but the velocities randomly assigned to the OLD,
    # high-energy positions stay attached to the particles regardless, so the very first
    # dynamics step combines brand-new (relaxed) positions/forces with stale, mismatched
    # velocities. Measured directly: for a real design whose starting energy was 6.4e10
    # kJ/mol, minimization alone reliably converged to a sane ~-9500 kJ/mol either way, but
    # dynamics only completed without a NaN blowup when velocities were assigned AFTER
    # minimization, on the already-relaxed structure -- with velocities-before-minimize
    # (the original order), the exact same minimized structure crashed with NaN every time.
    min_info = minimize(simulation, minimize_max_iterations)
    simulation.context.setVelocitiesToTemperature(temperature_K * kelvin, seed)

    dcd_path = out_dir / f"{tag}_trajectory.dcd"
    log_path = out_dir / f"{tag}_statedata.csv"
    from openmm.app import DCDReporter
    simulation.reporters.append(DCDReporter(str(dcd_path), report_interval_steps))
    simulation.reporters.append(StateDataReporter(
        str(log_path), report_interval_steps, step=True, time=True, potentialEnergy=True,
        kineticEnergy=True, temperature=True, speed=True,
    ))

    sps = probe_steps_per_second(simulation, n_probe_steps=100)
    target_steps = int((target_ns * 1e6) / timestep_fs)  # 1 ns = 1e6 fs
    est_wallclock = target_steps / sps if sps > 0 else float("inf")

    if est_wallclock > max_wallclock_s:
        affordable_steps = int(sps * max_wallclock_s)
        logger.warning(f"{tag}: target {target_ns} ns would take ~{est_wallclock:.0f}s at {sps:.1f} "
                        f"steps/s on this CPU; capping to a {max_wallclock_s:.0f}s budget "
                        f"({affordable_steps} steps) and logging the ACTUAL ns simulated, not the target.")
        remaining_steps = max(affordable_steps - 100, 0)
    else:
        remaining_steps = target_steps - 100

    start = time.time()
    if remaining_steps > 0:
        simulation.step(remaining_steps)
    elapsed = time.time() - start

    total_steps = 100 + max(remaining_steps, 0)
    actual_ns = (total_steps * timestep_fs) / 1e6

    final_state = simulation.context.getState(getEnergy=True, getPositions=True)
    final_pdb_path = out_dir / f"{tag}_final.pdb"
    with open(final_pdb_path, "w") as fh:
        PDBFile.writeFile(simulation.topology, final_state.getPositions(), fh)

    result = {
        "tag": tag, "input_pdb": str(pdb_path),
        "minimization": min_info,
        "measured_steps_per_second": round(sps, 2),
        "target_ns": target_ns, "actual_ns_simulated": round(actual_ns, 5),
        "total_steps": total_steps, "timestep_fs": timestep_fs,
        "wallclock_seconds_production": round(elapsed, 1),
        "temperature_K": temperature_K, "seed": seed,
        "trajectory_dcd": str(dcd_path), "statedata_csv": str(log_path),
        "final_structure_pdb": str(final_pdb_path),
        "capped_by_wallclock_budget": est_wallclock > max_wallclock_s,
    }
    logger.info(f"{tag}: simulated {actual_ns:.5f} ns ({total_steps} steps @ {timestep_fs} fs) "
                f"in {elapsed:.1f}s wall-clock ({sps:.1f} steps/s measured on this CPU).")
    return result
