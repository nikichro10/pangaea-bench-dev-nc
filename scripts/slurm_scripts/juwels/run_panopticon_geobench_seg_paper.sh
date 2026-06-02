#!/bin/bash

# -----------------------------------------------------------------------------
# CONFIGURATION
# Panopticon segmentation on GeoBench — paper-faithful setup.
#
# Matches the Panopticon paper as closely as possible (without aux decoder):
#   - Augmentation: HFlip(0.5) + VFlip(0.5) + RandomResizedCrop(scale=[0.8,1.0])
#   - LR scheduler: OneCycleLR (cosine, no warmup)
#   - LR sweep: [1e-3, 1e-4, 1e-5, 1e-6] — best val-mIoU selects winner
#   - n_epochs: 50
#   - batch_size: 50 per GPU × 4 GPUs = 200 effective (matches paper)
#   - finetune: false, decoder: seg_upernet, channels: 512
#
# Remaining differences vs. paper:
#   - No auxiliary FCNHead (paper: CE + 0.4*CE_aux)
#
# Overrides:
#   RUN_DATASETS="mnz-cattle"         run a subset for quick testing
#   RUN_LRS="1e-4 1e-5"              run a subset of LR values
# -----------------------------------------------------------------------------
readonly PROJECT_DIR="/p/project1/hai_1280/pangaea-bench-dev-nc"
readonly LOG_DIR="/p/project1/hai_1280/logs/panopticon_geobench_seg_paper"
readonly WORK_DIR="/p/project1/hai_1280/experiments/panopticon_geobench_seg_paper"
readonly GEO_BENCH_DIR="/p/project1/hai_1280/datasets/geobench"
readonly MICROMAMBA="/p/project1/hai_1280/micromamba/bin/micromamba"
readonly ENV_PATH="/p/project1/hai_1280/envs/pangaea-bench"

readonly ENCODER="panopticon"
readonly PREPROCESSING="seg_panopticon_paper_no_pad"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

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
    local -r lr=$4

    # effective_lr = base_lr × batch_size / 256 × num_gpus  (paper formula)
    # batch_size=200 (50 per GPU × 4), num_gpus=1 → scale = 200/256
    local -r effective_lr=$(python3 -c "print(f'{float(\"${lr}\") * 200 / 256:.6e}')")

    local -r lr_tag=$(echo "$lr" | sed 's/[eE]/-/g; s/\./-/g')
    local -r job_name="panopt_paper_${dataset}_lr${lr_tag}"
    local -r sbatch_script="${dataset}_panopticon_paper_lr${lr_tag}.sh"

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=4 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${ENCODER}"
    cmd+=" decoder=seg_upernet"
    cmd+=" preprocessing=${PREPROCESSING}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=segmentation"
    cmd+=" lr_scheduler=one_cycle_lr"
    cmd+=" task.trainer.n_epochs=50"
    cmd+=" batch_size=50"
    cmd+=" finetune=false"
    cmd+=" optimizer.lr=${effective_lr}"
    cmd+=" lr_scheduler.max_lr=${effective_lr}"
    cmd+=" work_dir=${WORK_DIR}/${dataset}/lr${lr_tag}"
    cmd+=" use_wandb=True"

    printf '\nLaunching [%s | base_lr=%s → effective_lr=%s]:\n%s\n' "$dataset" "$lr" "$effective_lr" "$cmd"

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
#SBATCH --output=${LOG_DIR}/${dataset}_lr${lr_tag}_%j.out
#SBATCH --error=${LOG_DIR}/${dataset}_lr${lr_tag}_%j.err
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
RUN_LRS="${RUN_LRS:-1e-3 1e-4 1e-5 1e-6}"

for dataset in "${!datasets[@]}"; do
    if [ "$RUN_DATASETS" != "all" ]; then
        [[ " $RUN_DATASETS " != *" $dataset "* ]] && continue
    fi
    read -r criterion time_limit <<< "${datasets[$dataset]}"
    for lr in $RUN_LRS; do
        launch_job "$dataset" "$criterion" "$time_limit" "$lr"
    done
done

echo "All jobs queued."
