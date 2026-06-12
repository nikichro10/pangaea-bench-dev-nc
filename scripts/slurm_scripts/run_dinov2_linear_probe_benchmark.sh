#!/bin/bash

# -----------------------------------------------------------------------------
# DINOv2 ViT-B/14 linear-probe benchmark  (pure-RGB baseline)
# Encoder frozen throughout; only a linear head is trained.
# See run_dinov2_mlp_fusion_benchmark.sh for RGB override rationale.
# -----------------------------------------------------------------------------

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/dinov2_linear_probe"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly WORK_DIR="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/dinov2_linear_probe"
readonly ENCODER="dinov2_vitb14"
readonly ENCODER_IMSIZE=224

readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c06s02"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

choose_preprocessing() {
    local -r dataset=$1
    local -r dataset_size=$2
    if [ "$dataset_size" -le "$ENCODER_IMSIZE" ]; then
        echo "seg_resize"
    elif [ "$dataset" = "mados" ]; then
        echo "seg_focus_crop"
    else
        echo "seg_default"
    fi
}

launch_job() {
    local -r dataset=$1
    local -r task=$2
    local -r decoder=$3
    local -r criterion=$4
    local -r time_limit=$5
    local -ra extra_args=("${@:6}")

    local -r base_dataset="${dataset%_optical}"
    local -r dataset_size="${dataset_sizes[$base_dataset]}"
    local -r preprocessing=$(choose_preprocessing "$base_dataset" "$dataset_size")
    local -r job_name="dinov2_lp_${dataset}"
    local -r sbatch_script="${dataset}_dinov2_lp.sh"

    trap 'rm -f "${sbatch_script}"' EXIT

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${ENCODER}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    cmd+=" dataset.root_path='${DATASET_PATH}/${base_dataset}'"
    cmd+=" work_dir=${WORK_DIR}"
    cmd+=" use_wandb=True"
    cmd+=" batch_size=32"
    cmd+=" optimizer.lr=4e-4"
    cmd+=" optimizer.weight_decay=0"

    if [ "${#extra_args[@]}" -gt 0 ]; then
        cmd+=" ${extra_args[*]}"
    fi

    printf '\nLaunching [%s] preprocessing=%s:\n%s\n' "$dataset" "$preprocessing" "$cmd"

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
#SBATCH --output=${LOG_DIR}/${dataset}_output.%j
#SBATCH --error=${LOG_DIR}/${dataset}_error.%j
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
    trap - EXIT
}

declare -A dataset_sizes=(
    ["hlsburnscars"]=512
    ["sen1floods11"]=512
    ["mados"]=240
    ["pastis"]=128
    ["dynamicen"]=1024
    ["croptypemapping"]=64
    ["spacenet7cd"]=256
    ["ai4smallfarms"]=496
    ["fivebillionpixels"]=520
)

declare -A rgb_override=(
    ["hlsburnscars"]="'encoder.input_bands={optical:[B2,B3,B4]}'"
    ["sen1floods11"]="'encoder.input_bands={optical:[B2,B3,B4]}'"
    ["mados"]="'encoder.input_bands={optical:[B2,B3,B4]}'"
    ["pastis"]="'encoder.input_bands={optical:[B2,B3,B4]}'"
    ["dynamicen"]="'encoder.input_bands={optical:[B2,B3,B4]}'"
    ["croptypemapping"]="'encoder.input_bands={optical:[B2,B3,B4]}'"
    ["spacenet7cd"]=""
    ["ai4smallfarms"]="'encoder.input_bands={optical:[B2,B3,B4]}'"
    ["fivebillionpixels"]="'encoder.input_bands={optical:[B4,B3,B2]}'"
)

# Format: "task decoder criterion time_limit [extra_args...]"
declare -A datasets=(
    ["hlsburnscars"]="segmentation    seg_linear_probe    cross_entropy  3:00:00"
    ["sen1floods11"]="segmentation    seg_linear_probe    cross_entropy  3:00:00"
    ["mados"]="segmentation          seg_linear_probe    cross_entropy  4:00:00"
    ["pastis"]="segmentation         seg_linear_probe    cross_entropy  6:00:00 task.evaluator.inference_mode=whole"
    ["dynamicen"]="segmentation      seg_linear_probe    cross_entropy  12:00:00"
    ["croptypemapping"]="segmentation seg_linear_probe    cross_entropy  6:00:00 task.evaluator.inference_mode=whole"
    ["spacenet7cd"]="change_detection seg_siam_linear_probe dice           6:00:00"
    ["ai4smallfarms"]="segmentation   seg_linear_probe    weighted_bce   6:00:00 task.trainer.best_metric_key=IoU data_replicate=2"
    ["fivebillionpixels"]="segmentation seg_linear_probe  cross_entropy  10:00:00"
)

RUN_DATASETS="all"

for dataset in "${!datasets[@]}"; do
    if [ "$RUN_DATASETS" != "all" ]; then
        [[ " $RUN_DATASETS " != *" $dataset "* ]] && continue
    fi
    read -r task decoder criterion time_limit remainder <<< "${datasets[$dataset]}"
    read -ra extra_args <<< "$remainder"
    override="${rgb_override[$dataset]}"
    [ -n "$override" ] && extra_args+=("$override")
    launch_job "$dataset" "$task" "$decoder" "$criterion" "$time_limit" "${extra_args[@]}"
done

echo "All jobs queued."
