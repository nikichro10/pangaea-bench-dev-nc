#!/bin/bash

# CARL normalization ablation: dataset-wise vs image-wise
# Launches 2 x 4 = 8 jobs.
#
#   imagewise : encoder.image_norm=true  (default) + *_no_norm preprocessing
#   dataset   : encoder.image_norm=false           + standard preprocessing (NormalizeMeanStd)

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/carl_norm_ablation"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly WORK_DIR="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/carl_norm_ablation"
readonly ENCODER="carl"
readonly ENCODER_IMSIZE=224

readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}"

declare -A dataset_sizes=(
    ["hlsburnscars"]=512
    ["sen1floods11"]=512
    ["mados"]=240
    ["fivebillionpixels"]=520
)

# Format: "decoder criterion time_limit"
declare -A dataset_cfg=(
    ["hlsburnscars"]="seg_upernet cross_entropy 6:00:00"
    ["sen1floods11"]="seg_upernet cross_entropy 6:00:00"
    ["mados"]="seg_upernet cross_entropy 8:00:00"
    ["fivebillionpixels"]="seg_upernet cross_entropy 10:00:00"
)

choose_base_preprocessing() {
    local -r dataset=$1
    local -r size=$2
    if [ "$size" -le "$ENCODER_IMSIZE" ]; then
        echo "seg_resize"
    elif [ "$dataset" = "mados" ]; then
        echo "seg_focus_crop"
    else
        echo "seg_default"
    fi
}

launch_job() {
    local -r dataset=$1
    local -r norm=$2   # "imagewise" or "dataset"
    local -r decoder=$3
    local -r criterion=$4
    local -r time_limit=$5

    local -r size="${dataset_sizes[$dataset]}"
    local -r base_preprocessing=$(choose_base_preprocessing "$dataset" "$size")

    local preprocessing image_norm_flag
    if [ "$norm" = "dataset" ]; then
        preprocessing="${base_preprocessing}"
        image_norm_flag="encoder.image_norm=false"
    else
        preprocessing="${base_preprocessing}_no_norm"
        image_norm_flag="encoder.image_norm=true"
    fi

    local -r job_name="carl_norm_${norm}_${dataset}"
    local -r sbatch_script="/tmp/${job_name}.sh"

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${ENCODER}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=segmentation"
    cmd+=" dataset.root_path='${DATASET_PATH}/${dataset}'"
    cmd+=" work_dir=${WORK_DIR}/${norm}"
    cmd+=" ${image_norm_flag}"
    cmd+=" use_wandb=True"
    cmd+=" wandb_project=carl-norm-ablation"
    cmd+=" task.trainer.ckpt_interval=999"

    printf 'Launching [%s] norm=%-10s preprocessing=%s\n' "$dataset" "$norm" "$preprocessing"

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
#SBATCH --time=${time_limit}
#SBATCH --exclude=${SLURM_EXCLUDE}
#SBATCH --output=${LOG_DIR}/${norm}_${dataset}_output.%j
#SBATCH --error=${LOG_DIR}/${norm}_${dataset}_error.%j
#SBATCH --chdir=/dss/dsshome1/08/di54zop/pangaea-bench_dev

export HYDRA_FULL_ERROR=1

module add micromamba
eval "\$(micromamba shell hook --shell bash)"
micromamba activate pangaea-bench

$cmd
EOF

    chmod +x "$sbatch_script"
    sbatch "$sbatch_script"
    rm "$sbatch_script"
}

for dataset in "${!dataset_cfg[@]}"; do
    read -r decoder criterion time_limit <<< "${dataset_cfg[$dataset]}"
    for norm in imagewise dataset; do
        launch_job "$dataset" "$norm" "$decoder" "$criterion" "$time_limit"
    done
done

echo "All 8 jobs queued."
