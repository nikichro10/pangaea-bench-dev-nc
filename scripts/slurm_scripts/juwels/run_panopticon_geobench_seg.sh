#!/bin/bash

# -----------------------------------------------------------------------------
# CONFIGURATION
# Panopticon segmentation on GeoBench optical datasets.
#
# Setup mirrors the PANGAEA paper baseline (run_dofa_geobench_seg_pangaea.sh):
#   - 4 GPUs, finetune=false
#
# Differs from DOFA PANGAEA only in preprocessing: seg_default_norm_no_pad
# instead of seg_default. Both have RandomCrop + NormalizeMeanStd; the only
# difference is that BandPadding is removed so run.py calls
# set_wv_list_from_data_bands and Panopticon sees only the actual bands.
#
# Overrides:
#   RUN_DATASETS="mcashew-plantation mnz-cattle"  run a subset for quick testing
#   RUN_DECODERS="seg_linear_probe"               run a single decoder
# -----------------------------------------------------------------------------
readonly PROJECT_DIR="/p/project1/hai_1280/pangaea-bench-dev-nc"
readonly LOG_DIR="/p/project1/hai_1280/logs/panopticon_geobench_seg"
readonly WORK_DIR="/p/project1/hai_1280/experiments/panopticon_geobench_seg"
readonly GEO_BENCH_DIR="/p/project1/hai_1280/datasets/geobench"
readonly MICROMAMBA="/p/project1/hai_1280/micromamba/bin/micromamba"
readonly ENV_PATH="/p/project1/hai_1280/envs/pangaea-bench"

readonly ENCODER="panopticon"
readonly PREPROCESSING="seg_default_norm_no_pad"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

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
# JOB LAUNCHER
# -----------------------------------------------------------------------------
launch_job() {
    local -r dataset=$1
    local -r decoder=$2
    local -r criterion=$3
    local -r time_limit=$4

    local -r job_name="panopt_seg_${decoder}_${dataset}"
    local -r sbatch_script="${dataset}_${decoder}_panopticon_seg.sh"

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=4 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${ENCODER}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${PREPROCESSING}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=segmentation"
    cmd+=" finetune=false"
    cmd+=" work_dir=${WORK_DIR}/${decoder}/${dataset}"
    cmd+=" use_wandb=True"

    printf '\nLaunching [%s | %s]:\n%s\n' "$dataset" "$decoder" "$cmd"

    trap 'rm -f "${sbatch_script}"' EXIT

    cat > "$sbatch_script" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --account=hai_1280
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=${time_limit}
#SBATCH --output=${LOG_DIR}/${decoder}_${dataset}_%j.out
#SBATCH --error=${LOG_DIR}/${decoder}_${dataset}_%j.err
#SBATCH --chdir=${PROJECT_DIR}

export HYDRA_FULL_ERROR=1
export GEO_BENCH_DIR="${GEO_BENCH_DIR}"
export MAMBA_ROOT_PREFIX="/p/project1/hai_1280/micromamba"
export WANDB_MODE=offline
export WANDB_PROJECT="Juwels"
export PYTHONPATH="${PROJECT_DIR}:\${PYTHONPATH}"

eval "\$(${MICROMAMBA} shell hook --shell bash)"
micromamba activate ${ENV_PATH}

$cmd
EOF

    chmod +x "$sbatch_script"
    sbatch "$sbatch_script"
    rm -f "$sbatch_script"
    trap - EXIT
}

# -----------------------------------------------------------------------------
# SUBMIT JOBS
# -----------------------------------------------------------------------------
RUN_DATASETS="${RUN_DATASETS:-all}"
RUN_DECODERS="${RUN_DECODERS:-seg_upernet seg_linear_probe}"

for dataset in "${!datasets[@]}"; do
    if [ "$RUN_DATASETS" != "all" ]; then
        [[ " $RUN_DATASETS " != *" $dataset "* ]] && continue
    fi

    read -r criterion time_limit <<< "${datasets[$dataset]}"

    for decoder in $RUN_DECODERS; do
        launch_job "$dataset" "$decoder" "$criterion" "$time_limit"
    done
done

echo "All jobs queued."
