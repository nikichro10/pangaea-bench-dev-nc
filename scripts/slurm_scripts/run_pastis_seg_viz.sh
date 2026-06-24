#!/bin/bash
# Generate segmentation visualizations for DOFA and DOFAv2 on PASTIS.
# Outputs one PNG per run to VIZ_DIR: Input | GT | Prediction grid.

readonly VIZ_DIR="/dss/dsshome1/08/di54zop/pangaea_experiments/pastis_seg_viz"
readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/pastis_seg_viz"
readonly BASE="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/pangaea_full_seg"

readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}"

declare -A RUNS=(
    ["dofa_upernet"]="${BASE}/upernet/dofa_seg/20260609_12207129_2fe2df_pastis_dofa_seg_upernet_mt_ltae_seg_resize"
    ["dofav2_upernet"]="${BASE}/upernet/dofav2_seg/20260609_12207138_80846a_pastis_dofav2_seg_upernet_mt_ltae_seg_resize"
    ["dofa_mlp"]="${BASE}/mlp_fusion/dofa_mlp_fusion/20260609_12205992_1420ab_pastis_dofa_seg_mlp_fusion_seg_resize"
    ["dofav2_mlp"]="${BASE}/mlp_fusion/dofav2_mlp_fusion/20260609_12207212_489010_pastis_dofav2_seg_mlp_fusion_seg_resize"
)

for label in "${!RUNS[@]}"; do
    ckpt_dir="${RUNS[$label]}"
    save_dir="${VIZ_DIR}/${label}"
    sbatch_script="${label}_viz.sh"

    trap 'rm -f "${sbatch_script}"' EXIT

    cat > "$sbatch_script" << EOF
#!/bin/bash
#SBATCH --job-name=viz_${label}
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
    echo "Submitting viz job for ${label} ..."
    sbatch "$sbatch_script"
    rm "$sbatch_script"
    trap - EXIT
done

echo "All viz jobs submitted. Output PNGs will appear in: ${VIZ_DIR}/"
