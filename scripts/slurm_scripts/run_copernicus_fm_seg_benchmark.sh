#!/bin/bash
# =============================================================================
# Copernicus-FM — Segmentation benchmark (UPerNet)
#
# Setup:
#   Backbone : ViT-B/16 (copernicus_fm_base), frozen encoder
#   Datasets : m-cashew-plantation, m-SA-crop-type
#   Decoder  : UPerNet
#   Optimizer: AdamW, LR 0.005, cosine decay, 20 epochs, batch size 64
#   Loss     : cross_entropy
#   Augment  : seg_geobench (random crop 224, rotation ±90°, H/V flip)
#   Metric   : mIoU
# =============================================================================

readonly ENCODER="copernicus_fm_base"
readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/copernicus_fm_seg"
readonly WORK_DIR="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/pangaea_full_seg/upernet/copernicus_fm_seg"
readonly GEO_BENCH="/p/project1/hai_1280/datasets/geobench"
readonly SLURM_PARTITION="hpda2_compute_gpu"
readonly SLURM_CLUSTER="hpda2"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s02,hpdar04c01s03"

mkdir -p "$LOG_DIR" "$WORK_DIR"

DATASETS=(mcashew-plantation msa-crop-type)

declare -A TIME_LIMIT=(
    [mcashew-plantation]="8:00:00"
    [msa-crop-type]="8:00:00"
)

for dataset in "${DATASETS[@]}"; do

    job_name="cfm_seg_${dataset}"
    sbatch_script="/tmp/${job_name}.sh"
    time_limit="${TIME_LIMIT[$dataset]}"

    cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${ENCODER}"
    cmd+=" decoder=upernet"
    cmd+=" preprocessing=seg_geobench"
    cmd+=" criterion=cross_entropy"
    cmd+=" task=segmentation"
    cmd+=" optimizer=adamw"
    cmd+=" lr_scheduler=cosine_annealing_lr"
    cmd+=" optimizer.lr=0.005"
    cmd+=" task.trainer.n_epochs=20"
    cmd+=" task.trainer.ckpt_interval=20"
    cmd+=" task.trainer.eval_interval=5"
    cmd+=" batch_size=64"
    cmd+=" finetune=false"
    cmd+=" work_dir=${WORK_DIR}"
    cmd+=" use_wandb=true"

    cat > "$sbatch_script" << SBATCH
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --clusters=${SLURM_CLUSTER}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --time=${time_limit}
#SBATCH --exclude=${SLURM_EXCLUDE}
#SBATCH --output=${LOG_DIR}/${dataset}_%j.out
#SBATCH --error=${LOG_DIR}/${dataset}_%j.err
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

    printf "Queued: %s  (UPerNet, AdamW lr=0.005, 20 epochs)\n" "$dataset"

done

echo ""
echo "Total jobs queued: ${#DATASETS[@]}"
