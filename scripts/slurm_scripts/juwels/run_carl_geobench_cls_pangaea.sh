#!/bin/bash

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
readonly PROJECT_DIR="/p/project1/hai_1280/pangaea-bench-dev-nc"
readonly LOG_DIR="/p/project1/hai_1280/logs/carl_geobench_cls_pangaea"
readonly WORK_DIR="/p/project1/hai_1280/experiments/carl_geobench_cls_pangaea"
readonly GEO_BENCH_DIR="/p/project1/hai_1280/datasets/geobench"
readonly MICROMAMBA="/p/project1/hai_1280/micromamba/bin/micromamba"
readonly ENV_PATH="/p/project1/hai_1280/envs/pangaea-bench"

readonly ENCODER="carl"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

# -----------------------------------------------------------------------------
# DATASET DEFINITIONS
# Format: "task criterion time_limit"
# mforestnet_l8 excluded: CARL's BAND_WAVELENGTHS only covers S2 band names.
# -----------------------------------------------------------------------------
declare -A datasets=(
    ["mbigearthnet"]="linear_classification_multi_label  binary_cross_entropy  1:00:00"
    ["mso2sat"]="linear_classification                   cross_entropy          1:00:00"
    ["mbrickkiln"]="linear_classification                cross_entropy          1:00:00"
    ["mforestnet"]="linear_classification                cross_entropy          1:00:00"
    ["meurosat"]="linear_classification                  cross_entropy          1:00:00"
    ["mpv4ger"]="linear_classification                   cross_entropy          1:00:00"
)

# -----------------------------------------------------------------------------
# JOB LAUNCHER
# -----------------------------------------------------------------------------
launch_job() {
    local -r dataset=$1
    local -r task=$2
    local -r criterion=$3
    local -r time_limit=$4

    local -r job_name="pangaea_cls_${dataset}"
    local -r sbatch_script="${dataset}_carl_pangaea_cls.sh"

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${ENCODER}"
    cmd+=" decoder=cls_linear"
    cmd+=" preprocessing=cls_resize"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    cmd+=" task.trainer.n_epochs=50"
    cmd+=" batch_size=16"
    cmd+=" finetune=false"
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
#SBATCH --gres=gpu:1
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
# SELECT DATASETS TO RUN
# "all" or space-separated list, e.g.: RUN_DATASETS="mforestnet meurosat"
# -----------------------------------------------------------------------------
RUN_DATASETS="${RUN_DATASETS:-all}"

# -----------------------------------------------------------------------------
# SUBMIT JOBS
# -----------------------------------------------------------------------------
for dataset in "${!datasets[@]}"; do
    if [ "$RUN_DATASETS" != "all" ]; then
        [[ " $RUN_DATASETS " != *" $dataset "* ]] && continue
    fi
    read -r task criterion time_limit <<< "${datasets[$dataset]}"
    launch_job "$dataset" "$task" "$criterion" "$time_limit"
done

echo "All jobs queued."
