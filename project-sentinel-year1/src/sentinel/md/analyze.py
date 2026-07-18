"""mdtraj-based trajectory analysis: RMSD, per-residue RMSF, radius of
gyration, and a secondary-structure timeline (mdtraj's built-in DSSP)."""
from __future__ import annotations

import mdtraj as md
import numpy as np


def load_trajectory(dcd_path: str, top_pdb_path: str) -> md.Trajectory:
    return md.load_dcd(dcd_path, top=top_pdb_path)


def compute_rmsd(traj: md.Trajectory, reference_frame: int = 0) -> list[float]:
    traj_ca = traj.atom_slice(traj.topology.select("name CA"))
    rmsd = md.rmsd(traj_ca, traj_ca, frame=reference_frame)
    return [float(x) for x in rmsd]


def compute_rmsf(traj: md.Trajectory) -> dict:
    traj_ca = traj.atom_slice(traj.topology.select("name CA"))
    traj_ca = traj_ca.superpose(traj_ca, frame=0)
    rmsf = md.rmsf(traj_ca, traj_ca, frame=0)
    residues = [traj_ca.topology.atom(i).residue for i in range(traj_ca.n_atoms)]
    return {
        "res_id": [r.resSeq for r in residues],
        "res_name": [r.name for r in residues],
        "chain": [str(r.chain.index) for r in residues],
        "rmsf_nm": [float(x) for x in rmsf],
    }


def compute_radius_of_gyration(traj: md.Trajectory) -> list[float]:
    rg = md.compute_rg(traj)
    return [float(x) for x in rg]


def compute_ss_timeline(traj: md.Trajectory) -> dict:
    """Per-frame, per-residue secondary structure via mdtraj's simplified DSSP."""
    ss = md.compute_dssp(traj, simplified=True)  # array [n_frames, n_residues] of 'H'/'E'/'C'
    n_frames, n_res = ss.shape
    beta_fraction_per_frame = [float(np.mean(ss[f] == "E")) for f in range(n_frames)]
    helix_fraction_per_frame = [float(np.mean(ss[f] == "H")) for f in range(n_frames)]
    return {
        "n_frames": n_frames, "n_residues": n_res,
        "beta_fraction_per_frame": beta_fraction_per_frame,
        "helix_fraction_per_frame": helix_fraction_per_frame,
        "ss_matrix": ss.tolist(),
    }


def full_analysis(dcd_path: str, top_pdb_path: str) -> dict:
    traj = load_trajectory(dcd_path, top_pdb_path)
    return {
        "n_frames": traj.n_frames,
        "rmsd_nm": compute_rmsd(traj),
        "rmsf": compute_rmsf(traj),
        "radius_of_gyration_nm": compute_radius_of_gyration(traj),
        "secondary_structure": compute_ss_timeline(traj),
    }
