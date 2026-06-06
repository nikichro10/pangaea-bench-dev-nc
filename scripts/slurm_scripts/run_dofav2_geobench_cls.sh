#!/bin/bash

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/dofav2_geobench_cls"
readonly WORK_DIR="/dss/dsshome1/08/di54zop/pangaea_experiments/dofav2_geobench_cls"
readonly ENCODER="dofav2"

# GEO-Bench root — dataset configs read ${oc.env:GEO_BENCH_DIR}/classification_v1.0/<dataset>
readonly GEO_BENCH_DIR="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/datasets/geobench"

readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c06s02"

# LR grid search as described in the DOFA+ paper.
# Best LR per dataset is selected based on validation accuracy.
readonly LR_GRID=(0.5 1.0 10 20)

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

# -----------------------------------------------------------------------------
# DATASET DEFINITIONS
# Format: "task criterion"
# mbigearthnet: multi-label → linear_classification_multi_label + multi_label_soft_margin
# all others:   cross-entropy classification
# -----------------------------------------------------------------------------
declare -A datasets=(
    ["mbigearthnet"]="linear_classification_multi_label multi_label_soft_margin"
    ["mso2sat"]="linear_classification                  cross_entropy"
    ["mbrickkiln"]="linear_classification               cross_entropy"
    ["mforestnet"]="linear_classification               cross_entropy"
    ["meurosat"]="linear_classification                 cross_entropy"
    ["mpv4ger"]="linear_classification                  cross_entropy"
)

# -----------------------------------------------------------------------------
# JOB LAUNCHER
# -----------------------------------------------------------------------------
launch_job() {
    local -r dataset=$1
    local -r task=$2
    local -r criterion=$3
    local -r lr=$4

    # Make LR filename-safe (0.5 -> 0p5)
    local -r lr_str="${lr//./p}"
    local -r job_name="dofav2cls_${dataset}_lr${lr_str}"
    local -r sbatch_script="${dataset}_lr${lr_str}_dofav2cls.sh"

    trap 'rm -f "${sbatch_script}"' EXIT

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${ENCODER}"
    cmd+=" decoder=cls_linear"
    cmd+=" preprocessing=cls_geobench"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    cmd+=" optimizer=lars"
    cmd+=" lr_scheduler=cosine_annealing_lr"
    cmd+=" optimizer.lr=${lr}"
    cmd+=" work_dir=${WORK_DIR}"
    cmd+=" use_wandb=True"
    cmd+=" task.trainer.ckpt_interval=999"

    printf '\nLaunching [%s] lr=%s:\n%s\n' "$dataset" "$lr" "$cmd"

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
#SBATCH --time=4:00:00
#SBATCH --exclude=${SLURM_EXCLUDE}
#SBATCH --output=${LOG_DIR}/${dataset}_lr${lr_str}_output.%j
#SBATCH --error=${LOG_DIR}/${dataset}_lr${lr_str}_error.%j
#SBATCH --chdir=/dss/dsshome1/08/di54zop/pangaea-bench_dev

export HYDRA_FULL_ERROR=1
export GEO_BENCH_DIR=${GEO_BENCH_DIR}

module add micromamba
eval "\$(micromamba shell hook --shell bash)"
micromamba activate pangaea-bench

$cmd
EOF

    chmod +x "$sbatch_script"
    sbatch "$sbatch_script"
    rm "$sbatch_script"
    trap - EXIT
}

# -----------------------------------------------------------------------------
# SELECT DATASETS TO RUN
# Use "all" or a space-separated list, e.g. "mbigearthnet meurosat"
# -----------------------------------------------------------------------------
RUN_DATASETS="all"

# -----------------------------------------------------------------------------
# SUBMIT JOBS — one per dataset × LR value (4 × 6 = 24 jobs total)
# -----------------------------------------------------------------------------
for dataset in "${!datasets[@]}"; do
    if [ "$RUN_DATASETS" != "all" ]; then
        [[ " $RUN_DATASETS " != *" $dataset "* ]] && continue
    fi

    read -r task criterion <<< "${datasets[$dataset]}"

    for lr in "${LR_GRID[@]}"; do
        launch_job "$dataset" "$task" "$criterion" "$lr"
    done
done

echo "All jobs queued."
