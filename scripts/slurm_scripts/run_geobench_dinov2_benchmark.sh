#!/bin/bash

# -----------------------------------------------------------------------------
# DINOv2 ViT-B/14 — GeoBench segmentation benchmark  (UPerNet + MLP-Fusion)
#
# DINOv2 is RGB-only (encoder.input_bands: optical: [B4, B3, B2]).
# All 6 GeoBench seg datasets contain B4, B3, B2 in their optical bands:
#   - mcashew-plantation  [B1,B2,B3,B4]  → BandFilter selects B4,B3,B2  ✓
#   - mchesapeake-landcover [B2,B3,B4,B8]→ BandFilter selects B4,B3,B2  ✓
#   - mneontree           [B4,B3,B2]     → exact match                   ✓
#   - mnz-cattle          [B4,B3,B2]     → exact match                   ✓
#   - mpv4ger-seg         [B4,B3,B2]     → exact match                   ✓
#   - msa-crop-type       [B1,B2,B3,B4]  → BandFilter selects B4,B3,B2  ✓
# No encoder.input_bands override is needed; BandFilter resolves by band name.
#
# All datasets have img_size > 224 → seg_default preprocessing
# (RandomCropToEncoder → 224, BandFilter, NormalizeMeanStd).
#
# Runs 2 decoders × 6 datasets = 12 jobs total.
# Results land in:
#   geobench/seg/UPerNet/dinov2/
#   geobench/seg/MLPFusion/dinov2/
#
# Override RUN_DECODERS / RUN_DATASETS at call time for partial runs:
#   RUN_DECODERS="upernet" RUN_DATASETS="mnz-cattle mpv4ger-seg" bash run_geobench_dinov2_benchmark.sh
# -----------------------------------------------------------------------------

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/geobench_dinov2"
readonly GEO_BENCH_DIR="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/datasets/geobench"
readonly BASE_WORK_DIR="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/geobench/seg"

readonly ENCODER="dinov2_vitb14"
readonly PREPROCESSING="seg_default"

readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}"

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
# DECODER DEFINITIONS
# Format: "hydra_decoder_config work_dir_subdir job_name_suffix log_suffix"
# -----------------------------------------------------------------------------
declare -A decoders=(
    ["upernet"]="seg_upernet  UPerNet    up   upernet"
    ["mlpfusion"]="seg_mlp_fusion MLPFusion mlp  mlpfusion"
)

# -----------------------------------------------------------------------------
# JOB LAUNCHER
# -----------------------------------------------------------------------------
launch_job() {
    local -r dec_key=$1
    local -r dec_config=$2
    local -r dec_workdir=$3
    local -r dec_jobtag=$4
    local -r dec_logtag=$5
    local -r dataset=$6
    local -r criterion=$7
    local -r time_limit=$8

    local -r work_dir="${BASE_WORK_DIR}/${dec_workdir}/dinov2"
    local -r job_name="dinov2_${dec_jobtag}_${dataset}"
    local -r sbatch_script="${dataset}_dinov2_${dec_logtag}.sh"

    mkdir -p "${work_dir}"

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${ENCODER}"
    cmd+=" decoder=${dec_config}"
    cmd+=" preprocessing=${PREPROCESSING}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=segmentation"
    cmd+=" work_dir=${work_dir}"
    cmd+=" use_wandb=True"
    cmd+=" wandb_project=geobench"

    printf '\nLaunching [dinov2 | %s | %s] dec=%s prep=%s:\n%s\n' \
        "$dec_key" "$dataset" "$dec_config" "$PREPROCESSING" "$cmd"

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
#SBATCH --output=${LOG_DIR}/dinov2_${dec_logtag}_${dataset}_output.%j
#SBATCH --error=${LOG_DIR}/dinov2_${dec_logtag}_${dataset}_error.%j
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
# SELECT DECODERS AND DATASETS TO RUN
# Set to "all" or a space-separated list of keys from the arrays above.
# -----------------------------------------------------------------------------
RUN_DECODERS="${RUN_DECODERS:-all}"
RUN_DATASETS="${RUN_DATASETS:-all}"

# -----------------------------------------------------------------------------
# SUBMIT JOBS
# -----------------------------------------------------------------------------
for dec_key in "${!decoders[@]}"; do
    if [ "$RUN_DECODERS" != "all" ]; then
        [[ " $RUN_DECODERS " != *" $dec_key "* ]] && continue
    fi

    read -r dec_config dec_workdir dec_jobtag dec_logtag <<< "${decoders[$dec_key]}"

    for dataset in "${!datasets[@]}"; do
        if [ "$RUN_DATASETS" != "all" ]; then
            [[ " $RUN_DATASETS " != *" $dataset "* ]] && continue
        fi

        read -r criterion time_limit <<< "${datasets[$dataset]}"

        launch_job "$dec_key" "$dec_config" "$dec_workdir" "$dec_jobtag" "$dec_logtag" \
            "$dataset" "$criterion" "$time_limit"
    done
done

echo "All jobs queued."
