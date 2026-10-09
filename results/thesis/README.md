# Thesis results snapshot (Chapter 5.1, multispectral)

Frozen record of every run behind the multispectral results of the Master's thesis
(PANGAEA + GeoBench). The hyperspectral part (Chapter 5.2) lives in `his-bench_dev`.

| File | Content |
|---|---|
| `runs.csv` | One row per run: thesis reference, experiment group, run dir, dataset, encoder, decoder, seed, finished flag, test metrics (mean IoU/F1/Precision/Recall or Accuracy) and per-class IoU |
| `configs/<group>/<run_dir>.yaml` | The resolved Hydra config of that run |

Regenerate with `python3 scripts/export_thesis_results.py`. It reads the final `[test]` block
of each `train.log-0` under the experiment root
`/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/` (full logs and checkpoints stay there).

Notes:
* Segmentation metrics are in percent; classification metrics (m-ForestNet) are fractions in [0, 1].
* Some encoder/dataset pairs have several runs (reruns, seeds). The analysis notebooks pick the
  run used in the thesis; `finished=False` marks runs without a final checkpoint.
* CARL uses the image-wise variant (`carl_imagewise*` folders) in all main results.

## Where each result comes from

All SLURM scripts are in `scripts/slurm_scripts/`, notebooks in `notebooks/`, figures in `notebooks/figures/`.

| Thesis | Experiment group(s) | SLURM scripts | Analysis |
|---|---|---|---|
| Table 7: PANGAEA, MLPFusion | `pangaea_full_seg/mlp_fusion` | `run_*_mlp_fusion_benchmark.sh` | `analyze_results.ipynb` |
| Table 8: PANGAEA, UPerNet | `pangaea_full_seg/upernet` | `run_*_seg_benchmark.sh`, `run_unet_seg_benchmark.sh` | `analyze_results.ipynb` |
| SN7-CD reruns used in Tables 7/8 | `ablations/spacenet7cd_weighted_bce*_fix` | `Ablations/run_spacenet7cd_weighted_bce*_fix.sh` | `analyze_results.ipynb` |
| Table 9: GeoBench segmentation | `geobench/seg/{MLPFusion,UPerNet}` | `run_geobench_seg_benchmark.sh`, `run_geobench_mlpfusion_benchmark.sh` | `analyze_results.ipynb` |
| Figure 14: qualitative examples | `pangaea_full_seg/upernet` (checkpoints) | `qualitative/submit_qualitative_*.sh` | `scripts/qualitative/plot_qualitative_grid.py` → `qualitative_grid*.pdf` |
| Figure 15: average ranks | as Tables 7–9 | – | `analyze_results.ipynb` |
| Table 10: normalization | `ablations/norm_ablation`, `carl_norm_ablation` | `Ablations/run_norm_ablation_*.sh`, `Ablations/run_carl_norm_ablation.sh` | `analyze_norm_ablation.ipynb` |
| Table 11: band padding | `ablations/band_padding` | `Ablations/run_band_padding_*.sh` | `analyze_band_padding.ipynb` |
| Table 12: RGB only (UPerNet) | `pangaea_rgb/upernet` | `rgb/run_*_rgb_seg_benchmark.sh` | `analyze_results.ipynb` |
| Figure 16: band degradation | `band_subset_degradation` | `band_degradation/run_*_band_degradation.sh` | `analyze_band_degradation.ipynb` → `band_degradation_lineplot.pdf` |
| Figure 17: S2 default vs. true vs. random λ | `wavelength_ablation*`, `p6_random_wl_ablation*`, `geobench/cls` (m-ForestNet) | `wavelength_ablation/run_wavelength_ablation*.sh`, `wavelength_ablation/run_p6_*.sh`, `run_geobench_cls_benchmark.sh` | `analyze_wrong_wavelength_summary.ipynb` → `wrong_wl_summary_cross_sensor_3way.pdf` |
| Figure 18: S2 default vs. true wavelengths | – (sensor metadata) | – | `analyze_wavelength_ablation.ipynb` → `wavelength_comparison.pdf` |
| Figure 19: corrupted wavelengths | `p1`–`p4_wlswap_ablation`, `p6_random_wl_ablation_upernet`, `wavelength_ablation_const_wl` | `wavelength_ablation/run_p{1,2,3,4}_wlswap_ablation.sh`, `run_p6_*.sh`, `run_const_wl_ablation.sh` | `analyze_wrong_wavelength_summary.ipynb` → `wrong_wl_summary_m_{cashew_plantation,sa_crop_type}_upernet.pdf` |

Random draws for the shuffle/random-λ conditions are fixed in
`scripts/slurm_scripts/wavelength_ablation/p{2,4,6}_*_draws.json`.
