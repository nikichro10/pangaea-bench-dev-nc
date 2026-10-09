# Legacy

Experiments and debugging material that are **not part of the Master's thesis results**.
Kept for reference; nothing in the main pipeline depends on these files.

| Item | Files | Notes |
|---|---|---|
| DOFAv2 ViT-L | `scripts/run_dofav2_vitl_seg_benchmark.sh` | Configs stay in `configs/encoder/dofav2_vitl*.yaml` (Hydra needs them there) |
| DOFA normalization experiment | `scripts/run_dofa_norm_experiment.sh` | Uses `configs/preprocessing/*_dofa_norm.yaml` |
| P5 alpha interpolation ablation | `scripts/run_p5_alpha_interp_ablation.sh`, `notebooks/analyze_p5_alpha_interp_ablation.ipynb`, `notebooks/figures/p5_*`, `p6_*` | Results in `experiments/p5_alpha_interp_ablation/`; the `p6_*` band-movement plots were made by this notebook |
| DOFA debugging | `notebooks/debug_dofa.ipynb` | |
| Early analysis scripts | `scripts/analyze_experiments/*.py` | LR / normalization analyses from before the thesis runs; the CSVs they read were deleted |

The scripts are run from the repo root as before, e.g. `bash legacy/scripts/run_dofa_norm_experiment.sh`.
The P5 notebook writes its figures to `legacy/notebooks/figures/` when run from `legacy/notebooks/`.
