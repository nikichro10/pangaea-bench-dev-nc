#!/bin/bash
# Generate segmentation visualizations for DOFA and DOFAv2 on MADOS.
# Outputs one PNG per run to VIZ_DIR: Input | GT | Prediction grid.

readonly VIZ_DIR="/dss/dsshome1/08/di54zop/pangaea_experiments/mados_seg_viz"
readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/mados_seg_viz"
readonly BASE="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/pangaea_full_seg"

readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}"

declare -A RUNS=(
    ["dofa_upernet"]="${BASE}/upernet/dofa_seg/20260609_12207135_8e5906_mados_dofa_seg_upernet_seg_focus_crop"
    ["dofav2_upernet"]="${BASE}/upernet/dofav2_seg/20260612_12222066_b718ab_mados_dofav2_seg_upernet_seg_focus_crop"
    ["dofa_mlp"]="${BASE}/mlp_fusion/dofa_mlp_fusion/20260609_12205998_b7f6f2_mados_dofa_seg_mlp_fusion_seg_focus_crop"
    ["dofav2_mlp"]="${BASE}/mlp_fusion/dofav2_mlp_fusion/20260609_12207218_a7b6f1_mados_dofav2_seg_mlp_fusion_seg_focus_crop"
)

for label in "${!RUNS[@]}"; do
    ckpt_dir="${RUNS[$label]}"
    save_dir="${VIZ_DIR}/${label}"
    sbatch_script="${label}_mados_viz.sh"

    trap 'rm -f "${sbatch_script}"' EXIT

    cat > "$sbatch_script" << EOF
#!/bin/bash
#SBATCH --job-name=viz_mados_${label}
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --clusters=hpda2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=1:00:00
#SBATCH --exclude=${SLURM_EXCLUDE}
#SBATCH --output=${LOG_DIR}/${label}_output.%j
#SBATCH --error=${LOG_DIR}/${label}_error.%j
#SBATCH --chdir=/dss/dsshome1/08/di54zop/pangaea-bench_dev

export HYDRA_FULL_ERROR=1

module add micromamba
eval "\$(micromamba shell hook --shell bash)"
micromamba activate pangaea-bench

torchrun --standalone --nnodes=1 --nproc_per_node=1 \\
    scripts/generate_seg_viz.py \\
    --ckpt_dir "${ckpt_dir}" \\
    --save_preds_dir "${save_dir}"
EOF

    chmod +x "$sbatch_script"
    echo "Submitting viz job for mados/${label} ..."
    sbatch "$sbatch_script"
    rm "$sbatch_script"
    trap - EXIT
done

echo "All jobs submitted. Output PNGs will appear in: ${VIZ_DIR}/"
