#!/bin/bash

# -----------------------------------------------------------------------------
# CONFIGURATION
# Panopticon segmentation on GeoBench optical datasets — SegConcatProbe decoder.
#
# SegConcatProbe is a lean multi-scale probe:
#   N encoder layers → 1×1 project → concat → 1×1+3×3 fusion → 1×1 cls → upsample
# Encoder is frozen (finetune=false). Sits between SegLinearProbe and UPerNet.
#
# Visualisations (input / GT / pred PNG grids) are saved every 20 epochs and
# after the final test to task.evaluator.save_preds_dir.
#
# Overrides:
#   RUN_DATASETS="mnz-cattle"               run a single dataset for quick testing
#   UPSAMPLE_MODE="bicubic"                 change upsampling interpolation
# -----------------------------------------------------------------------------
readonly PROJECT_DIR="/p/project1/hai_1280/pangaea-bench-dev-nc"
readonly LOG_DIR="/p/project1/hai_1280/logs/panopticon_geobench_seg_concat_probe"
readonly WORK_DIR="/p/project1/hai_1280/experiments/panopticon_geobench_seg_concat_probe"
readonly VIZ_DIR="/p/project1/hai_1280/experiments/panopticon_geobench_seg_concat_probe/viz"
readonly GEO_BENCH_DIR="/p/project1/hai_1280/datasets/geobench"
readonly MICROMAMBA="/p/project1/hai_1280/micromamba/bin/micromamba"
readonly ENV_PATH="/p/project1/hai_1280/envs/pangaea-bench"

readonly ENCODER="panopticon"
readonly PREPROCESSING="seg_default_norm_no_pad"
readonly UPSAMPLE_MODE="${UPSAMPLE_MODE:-bilinear}"

mkdir -p "${LOG_DIR}" "${WORK_DIR}" "${VIZ_DIR}"

# -----------------------------------------------------------------------------
# DATASET DEFINITIONS
# Format: "criterion time_limit"
# -----------------------------------------------------------------------------
declare -A datasets=(
    ["mcashew-plantation"]="cross_entropy  2:00:00"
    ["mchesapeake-landcover"]="cross_entropy  6:00:00"
    ["mneontree"]="cross_entropy  4:00:00"
    ["mnz-cattle"]="cross_entropy  1:00:00"
    ["mpv4ger-seg"]="cross_entropy  2:00:00"
    ["msa-crop-type"]="cross_entropy  2:00:00"
)

# -----------------------------------------------------------------------------
# JOB LAUNCHER
# -----------------------------------------------------------------------------
launch_job() {
    local -r dataset=$1
    local -r criterion=$2
    local -r time_limit=$3

    local -r job_name="panopt_cprobe_${dataset}"
    local -r sbatch_script="${dataset}_panopticon_concat_probe.sh"

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=4 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${ENCODER}"
    cmd+=" decoder=seg_mlp_fusion"
    cmd+=" preprocessing=${PREPROCESSING}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=segmentation"
    cmd+=" finetune=false"
    cmd+=" decoder.upsample_mode=${UPSAMPLE_MODE}"
    cmd+=" task.evaluator.save_preds_dir=${VIZ_DIR}/${dataset}/${UPSAMPLE_MODE}"
    cmd+=" work_dir=${WORK_DIR}/${dataset}"
    cmd+=" use_wandb=True"

    printf '\nLaunching [%s]:\n%s\n' "$dataset" "$cmd"

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
#SBATCH --output=${LOG_DIR}/${dataset}_%j.out
#SBATCH --error=${LOG_DIR}/${dataset}_%j.err
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

for dataset in "${!datasets[@]}"; do
    if [ "$RUN_DATASETS" != "all" ]; then
        [[ " $RUN_DATASETS " != *" $dataset "* ]] && continue
    fi
    read -r criterion time_limit <<< "${datasets[$dataset]}"
    launch_job "$dataset" "$criterion" "$time_limit"
done

echo "All jobs queued."
