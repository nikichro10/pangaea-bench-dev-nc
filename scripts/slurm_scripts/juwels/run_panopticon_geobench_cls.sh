#!/bin/bash

# -----------------------------------------------------------------------------
# CONFIGURATION
# Panopticon classification on GeoBench optical datasets.
#
# Setup mirrors the PANGAEA paper baseline (run_dofa_geobench_cls_pangaea.sh /
# run_carl_geobench_cls_pangaea.sh):
#   - Default optimizer (no LARS), batch_size=16, 1 GPU, n_epochs=50
#
# Differs from DOFA/CARL PANGAEA only in preprocessing: no BandPadding.
# Panopticon's channel cross-attention handles variable band sets natively;
# padding with zeros would add phantom channels at real wavelengths.
# run.py calls set_wv_list_from_data_bands to align wv_list with the
# dataset's actual bands.
#
# Override:
#   RUN_DATASETS="meurosat mforestnet"  run a subset for quick testing
# -----------------------------------------------------------------------------
readonly PROJECT_DIR="/p/project1/hai_1280/pangaea-bench-dev-nc"
readonly LOG_DIR="/p/project1/hai_1280/logs/panopticon_geobench_cls"
readonly WORK_DIR="/p/project1/hai_1280/experiments/panopticon_geobench_cls"
readonly GEO_BENCH_DIR="/p/project1/hai_1280/datasets/geobench"
readonly MICROMAMBA="/p/project1/hai_1280/micromamba/bin/micromamba"
readonly ENV_PATH="/p/project1/hai_1280/envs/pangaea-bench"

readonly ENCODER="panopticon"
readonly PREPROCESSING="cls_resize_norm_no_pad"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

# -----------------------------------------------------------------------------
# DATASET DEFINITIONS
# Format: "task criterion time_limit"
# -----------------------------------------------------------------------------
declare -A datasets=(
    ["mbigearthnet"]="linear_classification_multi_label  binary_cross_entropy  3:00:00"
    ["mso2sat"]="linear_classification                   cross_entropy          3:00:00"
    ["mbrickkiln"]="linear_classification                cross_entropy          3:00:00"
    ["mforestnet"]="linear_classification                cross_entropy          2:00:00"
    ["meurosat"]="linear_classification                  cross_entropy          1:00:00"
    ["mpv4ger"]="linear_classification                   cross_entropy          2:00:00"
)

# -----------------------------------------------------------------------------
# JOB LAUNCHER
# -----------------------------------------------------------------------------
launch_job() {
    local -r dataset=$1
    local -r task=$2
    local -r criterion=$3
    local -r time_limit=$4

    local -r job_name="panopt_cls_${dataset}"
    local -r sbatch_script="${dataset}_panopticon_cls.sh"

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${ENCODER}"
    cmd+=" decoder=cls_linear"
    cmd+=" preprocessing=${PREPROCESSING}"
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
# SUBMIT JOBS
# -----------------------------------------------------------------------------
RUN_DATASETS="${RUN_DATASETS:-all}"

for dataset in "${!datasets[@]}"; do
    if [ "$RUN_DATASETS" != "all" ]; then
        [[ " $RUN_DATASETS " != *" $dataset "* ]] && continue
    fi
    read -r task criterion time_limit <<< "${datasets[$dataset]}"
    launch_job "$dataset" "$task" "$criterion" "$time_limit"
done

echo "All jobs queued."
