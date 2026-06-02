#!/bin/bash

# -----------------------------------------------------------------------------
# CONFIGURATION
# Panopticon classification on 4 GeoBench datasets — implementation comparison
# against geobreeze.
#
# Matches geobreeze's linear-probe setup as closely as possible:
#   - Encoder blocks : [8, 9, 10, 11]  (matches geobreeze blk_indices.linear_probe)
#   - Optimizer      : SGD, momentum=0.9, weight_decay=0
#   - LR scheduler   : CosineAnnealingLR (eta_min=0)
#   - Batch size     : 200  (matches geobreeze)
#   - Epochs         : 50   (matches geobreeze)
#   - Augmentation   : Resize only — no random crops/flips (matches geobreeze no_aug)
#   - LR sweep       : same 13 base LRs as geobreeze, pre-scaled by bsz/256
#                      (geobreeze applies scale internally; we apply it here)
#
# Known differences vs geobreeze (acceptable for now):
#   - Head: pangaea uses AdaptiveAvgPool2d over patch tokens only;
#           geobreeze uses CLS + spatial-avg concat with 1D BatchNorm
#
# Datasets: meurosat, mforestnet, mpv4ger, mso2sat
# Jobs: 4 datasets × 13 LRs = 52 jobs total
#
# Usage:
#   bash run_panopticon_geobench_cls_eval.sh
#   RUN_DATASETS="meurosat mforestnet" bash run_panopticon_geobench_cls_eval.sh
# -----------------------------------------------------------------------------
readonly PROJECT_DIR="/p/project1/hai_1280/pangaea-bench-dev-nc"
readonly LOG_DIR="/p/project1/hai_1280/logs/panopticon_geobench_cls_eval"
readonly WORK_DIR="/p/project1/hai_1280/experiments/panopticon_geobench_cls_eval"
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
    ["meurosat"]="linear_classification   cross_entropy  1:00:00"
    ["mforestnet"]="linear_classification cross_entropy  2:00:00"
    ["mpv4ger"]="linear_classification    cross_entropy  2:00:00"
    ["mso2sat"]="linear_classification    cross_entropy  3:00:00"
)

# Effective LRs: geobreeze base_lrs * (200/256)
# base_lrs = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1, 5, 10]
readonly LRS="7.8125e-06 3.90625e-05 7.8125e-05 3.90625e-04 7.8125e-04 3.90625e-03 7.8125e-03 3.90625e-02 7.8125e-02 3.90625e-01 7.8125e-01 3.90625e+00 7.8125e+00"

# -----------------------------------------------------------------------------
# JOB LAUNCHER
# -----------------------------------------------------------------------------
launch_job() {
    local -r dataset=$1
    local -r task=$2
    local -r criterion=$3
    local -r time_limit=$4
    local -r lr=$5

    # Use a short tag for filenames (replace scientific notation)
    local -r lr_tag=$(python3 -c "print(f'{float(\"${lr}\"):.2e}'.replace('+','').replace('-0','-').replace('e0','e'))")
    local -r job_name="panopt_cls_eval_${dataset}_lr${lr_tag}"
    local -r sbatch_script="${dataset}_panopticon_cls_eval_lr${lr_tag}.sh"

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
    cmd+=" lr_scheduler=cosine_annealing_lr"
    cmd+=" task.trainer.n_epochs=50"
    cmd+=" batch_size=200"
    cmd+=" finetune=false"
    cmd+=" 'encoder.output_layers=[8,9,10,11]'"
    cmd+=" work_dir=${WORK_DIR}/${dataset}/lr${lr_tag}"
    cmd+=" use_wandb=True"

    printf '\nLaunching [%s | lr=%s]:\n%s\n' "$dataset" "$lr" "$cmd"

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
#SBATCH --output=${LOG_DIR}/${dataset}_lr${lr_tag}_%j.out
#SBATCH --error=${LOG_DIR}/${dataset}_lr${lr_tag}_%j.err
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
    read -r task criterion time_limit <<< "${datasets[$dataset]}"
    for lr in $LRS; do
        launch_job "$dataset" "$task" "$criterion" "$time_limit" "$lr"
    done
done

echo "All jobs queued."
