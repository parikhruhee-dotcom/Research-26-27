# MD scale-up: reproducing at full length / full GPU speed

This machine is a 2-core CPU sandbox with no GPU. Every MD run in this
module ran for real, but was capped to a wall-clock budget and (for the
fibril-tip system) truncated in size so it would finish. The exact
achieved simulation length for each system:

- **PHF6**: 0.1816 ns (target was 2.0 ns; capped_by_wallclock_budget=True)
- **PHF6_star**: 0.20335 ns (target was 2.0 ns; capped_by_wallclock_budget=True)
- **fibril_tip**: 0.0027 ns (target was 1.0 ns; capped_by_wallclock_budget=True)

## To reproduce at full scale on a GPU workstation

```bash
python -c "
from sentinel.md.run_md import run_hexapeptide_system, run_fibril_tip_system
from sentinel.utils.config import load_config, repo_path
cfg = load_config()
# 1. Edit config.yaml: md.hexapeptide.target_ns_cpu / md.fibril_tip.target_ns_cpu to the
#    desired full length (e.g. 100-500 ns), and set a CUDA platform in simulate.build_simulation
#    (Platform.getPlatformByName('CUDA') instead of 'CPU').
# 2. In run_md.run_fibril_tip_system, use all layers in the M1d *_stack.pdb (not a 2-layer
#    truncation) and raise MAX_WALLCLOCK_FIBRIL_TIP_S / drop the wall-clock cap entirely.
run_hexapeptide_system('PHF6', cfg, repo_path('results','md'))
run_fibril_tip_system(cfg, repo_path('results','md'))
"
```

No GPU-specific model weights are needed for this module (OpenMM MD runs
identically on CPU/GPU, only faster) — there is no Colab notebook for M4;
the code above is the full scale-up path.