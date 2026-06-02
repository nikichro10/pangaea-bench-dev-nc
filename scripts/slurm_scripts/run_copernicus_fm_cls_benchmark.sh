#!/bin/bash
# =============================================================================
# Copernicus-FM — Classification benchmark
#
# Setup:
#   Backbone : ViT-B/16 (copernicus_fm_base), frozen encoder
#   Datasets : m-bigearthnet (multi-label), m-eurosat,
#              m-cashew-plantation (pixel linear probe), m-SA-crop-type (pixel linear probe)
#   Decoder  : cls_linear (classification) or seg_linear_probe (dense datasets)
#   Optimizer: LARS, cosine decay, 50 epochs
#   LR grid  : 0.5 1.0 10 20  (one job per LR)
#   Loss     : multi_label_soft_margin (bigearthnet) | cross_entropy (others)
#   Augment  : cls_geobench (classification), seg_geobench (dense)
# =============================================================================

readonly ENCODER="copernicus_fm_base"
readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/copernicus_fm_cls"
readonly WORK_DIR="/dss/dsshome1/08/di54zop/pangaea_experiments/copernicus_fm_cls"
readonly GEO_BENCH="/p/project1/hai_1280/datasets/geobench"
readonly SLURM_PARTITION="hpda2_compute_gpu"
readonly SLURM_CLUSTER="hpda2"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s02,hpdar04c01s03"

mkdir -p "$LOG_DIR" "$WORK_DIR"

# ---------------------------------------------------------------------------
# Dataset table:
#   dataset_key  task                         decoder           criterion                  preprocessing  n_epochs
# ---------------------------------------------------------------------------
declare -A TASK=(
    [mbigearthnet]="linear_classification_multi_label"
    [meurosat]="linear_classification"
    [mcashew-plantation]="segmentation"
    [msa-crop-type]="segmentation"
)
declare -A DECODER=(
    [mbigearthnet]="cls_linear"
    [meurosat]="cls_linear"
    [mcashew-plantation]="seg_linear_probe"
    [msa-crop-type]="seg_linear_probe"
)
declare -A CRITERION=(
    [mbigearthnet]="multi_label_soft_margin"
    [meurosat]="cross_entropy"
    [mcashew-plantation]="cross_entropy"
    [msa-crop-type]="cross_entropy"
)
declare -A PREPROCESSING=(
    [mbigearthnet]="cls_geobench"
    [meurosat]="cls_geobench"
    [mcashew-plantation]="seg_geobench"
    [msa-crop-type]="seg_geobench"
)
declare -A N_EPOCHS=(
    [mbigearthnet]="50"
    [meurosat]="50"
    [mcashew-plantation]="50"
    [msa-crop-type]="50"
)
declare -A TIME_LIMIT=(
    [mbigearthnet]="4:00:00"
    [meurosat]="2:00:00"
    [mcashew-plantation]="6:00:00"
    [msa-crop-type]="6:00:00"
)

LR_VALUES=(0.5 1.0 10 20)
DATASETS=(mbigearthnet meurosat mcashew-plantation msa-crop-type)

# ---------------------------------------------------------------------------
total=0
for dataset in "${DATASETS[@]}"; do
for lr in "${LR_VALUES[@]}"; do

    lr_tag="${lr//./_}"   # e.g. 1.0 -> 1_0 for safe job names
    job_name="cfm_cls_${dataset}_lr${lr_tag}"
    sbatch_script="/tmp/${job_name}.sh"

    task="${TASK[$dataset]}"
    decoder="${DECODER[$dataset]}"
    criterion="${CRITERION[$dataset]}"
    preprocessing="${PREPROCESSING[$dataset]}"
    n_epochs="${N_EPOCHS[$dataset]}"
    time_limit="${TIME_LIMIT[$dataset]}"

    # For segmentation-style linear probe, override n_epochs inside task
    if [[ "$task" == "segmentation" ]]; then
        epoch_override="task.trainer.n_epochs=${n_epochs}"
    else
        epoch_override="task.trainer.n_epochs=${n_epochs}"
    fi

    cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${ENCODER}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    cmd+=" optimizer=lars"
    cmd+=" lr_scheduler=cosine_annealing_lr"
    cmd+=" optimizer.lr=${lr}"
    cmd+=" ${epoch_override}"
    cmd+=" finetune=false"
    cmd+=" batch_size=64"
    cmd+=" work_dir=${WORK_DIR}"
    cmd+=" use_wandb=true"
    cmd+=" task.trainer.ckpt_interval=999"

    cat > "$sbatch_script" << SBATCH
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --clusters=${SLURM_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=${time_limit}
#SBATCH --exclude=${SLURM_EXCLUDE}
#SBATCH --output=${LOG_DIR}/${dataset}_lr${lr_tag}_%j.out
#SBATCH --error=${LOG_DIR}/${dataset}_lr${lr_tag}_%j.err
#SBATCH --chdir=/dss/dsshome1/08/di54zop/pangaea-bench_dev

export HYDRA_FULL_ERROR=1
export GEO_BENCH_DIR=${GEO_BENCH}

module add micromamba
eval "\$(micromamba shell hook --shell bash)"
micromamba activate pangaea-bench

${cmd}
SBATCH

    chmod +x "$sbatch_script"
    sbatch "$sbatch_script"
    rm "$sbatch_script"

    printf "Queued: %-45s  lr=%-4s  task=%s\n" "$dataset" "$lr" "$task"
    (( total++ ))
done
done

echo ""
echo "Total jobs queued: ${total}  (${#DATASETS[@]} datasets × ${#LR_VALUES[@]} LR values)"
