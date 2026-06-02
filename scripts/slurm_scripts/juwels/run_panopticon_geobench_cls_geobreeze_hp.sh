#!/bin/bash

# -----------------------------------------------------------------------------
# CONFIGURATION
# Panopticon classification on GeoBench datasets using the best hyperparameters
# found by the geobreeze exhaustive linear-probe sweep:
#
#   Optimizer : SGD, momentum=0.9, weight_decay=0  (matches geobreeze)
#   Batch size: 200                                 (matches geobreeze)
#   LR        : per-dataset best base-lr from geobreeze sweep
#
# Best configs from geobreeze (batch_size=200, scale_lr = base_lr * 200/256):
#   m-eurosat    : blocks=4, pooling=avgpool, base_lr=1.0
#   m-brick-kiln : blocks=4, pooling=cls,     base_lr=0.005
#   m-pv4ger     : blocks=1, pooling=avgpool, base_lr=0.001
#   m-forestnet  : blocks=4, pooling=avgpool, base_lr=0.001
#   m-so2sat     : blocks=4, pooling=avgpool, base_lr=0.005
#   m-bigearthnet: no geobreeze result → using 0.001 as default
#
# Note: pangaea's LinearClassifier uses spatial avgpool over concatenated
# multi-block feature maps, which differs from geobreeze's token-level pooling.
# m-brick-kiln used CLS pooling in geobreeze; pangaea cannot replicate this
# without decoder changes, so avgpool is used for all datasets here.
# -----------------------------------------------------------------------------
readonly PROJECT_DIR="/p/project1/hai_1280/pangaea-bench-dev-nc"
readonly LOG_DIR="/p/project1/hai_1280/logs/panopticon_geobench_cls_geobreeze_hp"
readonly WORK_DIR="/p/project1/hai_1280/experiments/panopticon_geobench_cls_geobreeze_hp"
readonly GEO_BENCH_DIR="/p/project1/hai_1280/datasets/geobench"
readonly MICROMAMBA="/p/project1/hai_1280/micromamba/bin/micromamba"
readonly ENV_PATH="/p/project1/hai_1280/envs/pangaea-bench"

readonly ENCODER="panopticon"
readonly PREPROCESSING="cls_resize_norm_no_pad"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

# -----------------------------------------------------------------------------
# DATASET DEFINITIONS
# Format: "task criterion time_limit best_lr"
# -----------------------------------------------------------------------------
declare -A datasets=(
    ["meurosat"]="linear_classification    cross_entropy          1:00:00  1.0"
    ["mbrickkiln"]="linear_classification  cross_entropy          3:00:00  0.005"
    ["mpv4ger"]="linear_classification     cross_entropy          2:00:00  0.001"
    ["mforestnet"]="linear_classification  cross_entropy          2:00:00  0.001"
    ["mso2sat"]="linear_classification     cross_entropy          3:00:00  0.005"
    ["mbigearthnet"]="linear_classification_multi_label  binary_cross_entropy  3:00:00  0.001"
)

# -----------------------------------------------------------------------------
# JOB LAUNCHER
# -----------------------------------------------------------------------------
launch_job() {
    local -r dataset=$1
    local -r task=$2
    local -r criterion=$3
    local -r time_limit=$4
    local -r lr=$5

    local -r job_name="panopt_gb_${dataset}"
    local -r sbatch_script="${dataset}_panopticon_cls_geobreeze_hp.sh"

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${ENCODER}"
    cmd+=" decoder=cls_linear"
    cmd+=" preprocessing=${PREPROCESSING}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    cmd+=" optimizer=sgd"
    cmd+=" optimizer.momentum=0.9"
    cmd+=" optimizer.weight_decay=0"
    cmd+=" optimizer.lr=${lr}"
    cmd+=" task.trainer.n_epochs=50"
    cmd+=" batch_size=200"
    cmd+=" finetune=false"
    cmd+=" work_dir=${WORK_DIR}/${dataset}"
    cmd+=" use_wandb=True"

    printf '\nLaunching [%s] with lr=%s:\n%s\n' "$dataset" "$lr" "$cmd"

    trap 'rm -f "${sbatch_script}"' EXIT

    cat > "$sbatch_script" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --account=hai_1280
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --time=${time_limit}
#SBATCH --output=${LOG_DIR}/${dataset}_%j.out
#SBATCH --error=${LOG_DIR}/${dataset}_%j.err
#SBATCH --chdir=${PROJECT_DIR}

export HYDRA_FULL_ERROR=1
export GEO_BENCH_DIR="${GEO_BENCH_DIR}"
export TORCH_HOME="/p/project1/hai_1280/hf_cache"
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
    read -r task criterion time_limit lr <<< "${datasets[$dataset]}"
    launch_job "$dataset" "$task" "$criterion" "$time_limit" "$lr"
done

echo "All jobs queued."
