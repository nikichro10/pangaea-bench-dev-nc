#!/bin/bash

# -----------------------------------------------------------------------------
# GeoBench segmentation MLP-Fusion benchmark — hpda2
#
# Mirror of run_geobench_seg_benchmark.sh with seg_mlp_fusion decoder instead
# of seg_upernet. UNet is excluded (it uses seg_unet and has no mlp_fusion
# variant, consistent with pangaea_full_seg/mlp_fusion).
#
# 10 encoder runs × 6 GeoBench seg datasets (all single-temporal, img_size > 224).
#
# All encoders use seg_default (dataset-wise NormalizeMeanStd), matching the
# pangaea_full_seg setup. CARL has a second run with imagewise normalization:
#   carl          — carl + seg_default               (standard, matches pangaea_full_seg)
#   carl_imagewise — carl_no_internal_norm + seg_default_imagewise_norm
# Both CARL variants are frozen (finetune=False).
#
# Dataset root is controlled via GEO_BENCH_DIR env var, which the dataset
# YAML configs resolve at runtime (root_path: ${oc.env:GEO_BENCH_DIR}/segmentation_v1.0/...).
#
# Override RUN_ENCODERS / RUN_DATASETS at call time for partial runs:
#   RUN_ENCODERS="dofa carl carl_imagewise" RUN_DATASETS="mnz-cattle" bash run_geobench_mlpfusion_benchmark.sh
# -----------------------------------------------------------------------------

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/geobench_mlpfusion"
readonly GEO_BENCH_DIR="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/datasets/geobench"
readonly BASE_WORK_DIR="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/geobench/seg/MLPFusion"

readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}" "${BASE_WORK_DIR}"

# -----------------------------------------------------------------------------
# DATASET DEFINITIONS
# Format: "criterion time_limit"
# -----------------------------------------------------------------------------
declare -A datasets=(
    ["mcashew-plantation"]="cross_entropy  6:00:00"
    ["mchesapeake-landcover"]="cross_entropy  6:00:00"
    ["mneontree"]="cross_entropy  8:00:00"
    ["mnz-cattle"]="cross_entropy  6:00:00"
    ["mpv4ger-seg"]="cross_entropy  6:00:00"
    ["msa-crop-type"]="cross_entropy  6:00:00"
)

# -----------------------------------------------------------------------------
# ENCODER DEFINITIONS
# Format: "encoder_config decoder preprocessing [extra_hydra_args...]"
# Keys are used as work_dir subdirectory names and SLURM job name prefixes.
# -----------------------------------------------------------------------------
declare -A encoders=(
    ["dofa"]="dofa                    seg_mlp_fusion seg_default"
    ["dofav2"]="dofav2               seg_mlp_fusion seg_default"
    ["carl"]="carl                    seg_mlp_fusion seg_default                finetune=False"
    ["carl_imagewise"]="carl_no_internal_norm seg_mlp_fusion seg_default_imagewise_norm finetune=False"
    ["panopticon"]="panopticon        seg_mlp_fusion seg_default"
    ["smarties_vitb"]="smarties_vitb  seg_mlp_fusion seg_default"
    ["copernicus_fm_base"]="copernicus_fm_base seg_mlp_fusion seg_default"
    ["dinov2"]="dinov2_vitb14            seg_mlp_fusion seg_default"
    ["dinov3_vitl"]="dinov3_vitl16_sat493m seg_mlp_fusion seg_default"
    ["vit_scratch"]="vit_scratch      seg_mlp_fusion seg_default                finetune=True"
    ["vit_frozen_p16"]="vit_scratch   seg_mlp_fusion seg_default                finetune=False"
    ["vit_frozen_p8"]="vit_scratch_p8 seg_mlp_fusion seg_default                finetune=False"
)

# -----------------------------------------------------------------------------
# JOB LAUNCHER
# -----------------------------------------------------------------------------
launch_job() {
    local -r enc_key=$1
    local -r enc_config=$2
    local -r decoder=$3
    local -r preprocessing=$4
    local -r dataset=$5
    local -r criterion=$6
    local -r time_limit=$7
    local -ra extra_args=("${@:8}")

    local -r work_dir="${BASE_WORK_DIR}/${enc_key}"
    local -r job_name="${enc_key}_mlp_${dataset}"
    local -r sbatch_script="${dataset}_${enc_key}_geobench_mlp.sh"

    mkdir -p "${work_dir}"

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${enc_config}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=segmentation"
    cmd+=" work_dir=${work_dir}"
    cmd+=" use_wandb=True"
    cmd+=" wandb_project=geobench"

    if [ "${#extra_args[@]}" -gt 0 ]; then
        cmd+=" ${extra_args[*]}"
    fi

    printf '\nLaunching [%s | %s] enc=%s dec=%s prep=%s:\n%s\n' \
        "$enc_key" "$dataset" "$enc_config" "$decoder" "$preprocessing" "$cmd"

    trap 'rm -f "${sbatch_script}"' EXIT

    cat > "$sbatch_script" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --clusters=hpda2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=${time_limit}
#SBATCH --exclude=${SLURM_EXCLUDE}
#SBATCH --output=${LOG_DIR}/${enc_key}_${dataset}_output.%j
#SBATCH --error=${LOG_DIR}/${enc_key}_${dataset}_error.%j
#SBATCH --chdir=/dss/dsshome1/08/di54zop/pangaea-bench_dev

export HYDRA_FULL_ERROR=1
export GEO_BENCH_DIR="${GEO_BENCH_DIR}"

module add micromamba
eval "\$(micromamba shell hook --shell bash)"
micromamba activate pangaea-bench

$cmd
EOF

    chmod +x "$sbatch_script"
    sbatch "$sbatch_script"
    rm -f "$sbatch_script"
    trap - EXIT
}

# -----------------------------------------------------------------------------
# SELECT ENCODERS AND DATASETS TO RUN
# Set to "all" or a space-separated list of keys from the arrays above.
# Examples:
#   RUN_ENCODERS="dofa panopticon smarties_vitb"
#   RUN_DATASETS="mcashew-plantation mnz-cattle"
# -----------------------------------------------------------------------------
RUN_ENCODERS="${RUN_ENCODERS:-all}"
RUN_DATASETS="${RUN_DATASETS:-all}"

# -----------------------------------------------------------------------------
# SUBMIT JOBS
# -----------------------------------------------------------------------------
for enc_key in "${!encoders[@]}"; do
    if [ "$RUN_ENCODERS" != "all" ]; then
        [[ " $RUN_ENCODERS " != *" $enc_key "* ]] && continue
    fi

    read -r enc_config decoder preprocessing remainder <<< "${encoders[$enc_key]}"
    read -ra enc_extra_args <<< "$remainder"

    for dataset in "${!datasets[@]}"; do
        if [ "$RUN_DATASETS" != "all" ]; then
            [[ " $RUN_DATASETS " != *" $dataset "* ]] && continue
        fi

        read -r criterion time_limit <<< "${datasets[$dataset]}"

        launch_job "$enc_key" "$enc_config" "$decoder" "$preprocessing" \
            "$dataset" "$criterion" "$time_limit" "${enc_extra_args[@]}"
    done
done

echo "All jobs queued."
