#!/bin/bash

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
readonly PROJECT_DIR="/p/project1/hai_1280/pangaea-bench-dev-nc"
readonly LOG_DIR="/p/project1/hai_1280/logs/dofa_geobench_cls"
readonly WORK_DIR="/p/project1/hai_1280/experiments/dofa_geobench_cls"
readonly GEO_BENCH_DIR="/p/project1/hai_1280/datasets/geobench"
readonly MICROMAMBA="/p/project1/hai_1280/micromamba/bin/micromamba"
readonly ENV_PATH="/p/project1/hai_1280/envs/pangaea-bench"

readonly ENCODER="dofa"
readonly PREPROCESSING="cls_geobench"

# Learning rate grid from the DOFA paper: [0.5, 1.0, 10, 20]
# Set LR_GRID="all" to submit one job per LR, or a single value e.g. LR_GRID="1.0"
LR_GRID="${LR_GRID:-all}"

# Optimizer: LARS with cosine decay, as used in the DOFA paper.
# LR grid search over [0.5, 1.0, 10, 20] per dataset (set LR_GRID="all" or a single value).

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

# -----------------------------------------------------------------------------
# DATASET DEFINITIONS
# Format: "task decoder criterion time_limit batch_size"
# mbigearthnet is multi-label -> different task and criterion
# -----------------------------------------------------------------------------
declare -A datasets=(
    ["mbigearthnet"]="linear_classification_multi_label  cls_linear  binary_cross_entropy  1:00:00  64"
    ["mso2sat"]="linear_classification                   cls_linear  cross_entropy          1:00:00  64"
    ["mbrickkiln"]="linear_classification                cls_linear  cross_entropy          1:00:00  64"
    ["mforestnet"]="linear_classification                cls_linear  cross_entropy          1:00:00  64"
    ["meurosat"]="linear_classification                  cls_linear  cross_entropy          1:00:00  64"
    ["mpv4ger"]="linear_classification                   cls_linear  cross_entropy          1:00:00  64"
)

# -----------------------------------------------------------------------------
# JOB LAUNCHER
# -----------------------------------------------------------------------------
launch_job() {
    local -r dataset=$1
    local -r task=$2
    local -r decoder=$3
    local -r criterion=$4
    local -r time_limit=$5
    local -r batch_size=$6
    local -r lr=$7

    local -r job_name="dofa_cls_${dataset}_lr${lr}"
    local -r sbatch_script="${dataset}_${lr}_dofa_cls.sh"

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=4 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${ENCODER}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${PREPROCESSING}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    cmd+=" optimizer=lars"
    cmd+=" optimizer.lr=${lr}"
    cmd+=" lr_scheduler=cosine_annealing_lr"
    cmd+=" task.trainer.n_epochs=50"
    cmd+=" batch_size=${batch_size}"
    cmd+=" work_dir=${WORK_DIR}/${dataset}/lr${lr}"
    cmd+=" use_wandb=True"

    printf '\nLaunching [%s] lr=%s:\n%s\n' "$dataset" "$lr" "$cmd"

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
#SBATCH --output=${LOG_DIR}/${dataset}_lr${lr}_%j.out
#SBATCH --error=${LOG_DIR}/${dataset}_lr${lr}_%j.err
#SBATCH --chdir=${PROJECT_DIR}

export HYDRA_FULL_ERROR=1
export GEO_BENCH_DIR="${GEO_BENCH_DIR}"
export MAMBA_ROOT_PREFIX="/p/project1/hai_1280/micromamba"
export WANDB_MODE=offline
export WANDB_PROJECT="Juwels"

eval "\$(${MICROMAMBA} shell hook --shell bash)"
micromamba activate ${ENV_PATH}

export PYTHONPATH="${PROJECT_DIR}:\${PYTHONPATH}"

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
lr_values=(0.5 1.0 10 20)

for dataset in "${!datasets[@]}"; do
    read -r task decoder criterion time_limit batch_size <<< "${datasets[$dataset]}"

    if [ "$LR_GRID" = "all" ]; then
        for lr in "${lr_values[@]}"; do
            launch_job "$dataset" "$task" "$decoder" "$criterion" "$time_limit" "$batch_size" "$lr"
        done
    else
        launch_job "$dataset" "$task" "$decoder" "$criterion" "$time_limit" "$batch_size" "$LR_GRID"
    fi
done

echo "All jobs queued."
