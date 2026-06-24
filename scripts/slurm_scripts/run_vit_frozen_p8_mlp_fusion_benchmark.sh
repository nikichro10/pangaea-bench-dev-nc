#!/bin/bash

# -----------------------------------------------------------------------------
# ViT-B/16 patch-8x8, random-init, frozen encoder, MLP fusion benchmark
#
# Same as run_vit_scratch_frozen_encoder_mlp_fusion_benchmark.sh but with
# patch_size=8 (224px input → 28×28=784 tokens vs 14×14=196 for patch16).
# Encoder weights are randomly initialized and kept frozen — only the MLP
# fusion decoder is trained. Patch-size ablation counterpart to the UPerNet
# version (run_vit_scratch_p8_frozen_encoder_seg_benchmark.sh).
#
# All datasets use vit_scratch_p8 (MLP fusion does not require the MT variant).
# Single/multi-temporal segmentation uses seg_mlp_fusion.
# Change detection (spacenet7cd) uses seg_siam_mlp_fusion.
#
# Input size: 224px. finetune=False (encoder frozen).
# -----------------------------------------------------------------------------

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/vit_scratch_p8_frozen_encoder_mlp_fusion"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly WORK_DIR="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/pangaea_full_seg/mlp_fusion/vit_scratch_p8_frozen_encoder_mlp_fusion"
readonly ENCODER_IMSIZE=224

readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

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
    local -r encoder=$3
    local -r decoder=$4
    local -r criterion=$5
    local -r time_limit=$6
    local -ra extra_args=("${@:7}")

    local -r dataset_size="${dataset_sizes[$dataset]}"
    local -r preprocessing=$(choose_preprocessing "$dataset" "$dataset_size")
    local -r job_name="vit_p8_frozen_mlp_${dataset}"
    local -r sbatch_script="${dataset}_vit_p8_frozen_mlp.sh"

    trap 'rm -f "${sbatch_script}"' EXIT

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${encoder}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    cmd+=" dataset.root_path='${DATASET_PATH}/${dataset}'"
    cmd+=" work_dir=${WORK_DIR}"
    cmd+=" use_wandb=True"
    cmd+=" wandb_project=geofm-bench"
    cmd+=" finetune=False"

    if [ "${#extra_args[@]}" -gt 0 ]; then
        cmd+=" ${extra_args[*]}"
    fi

    printf '\nLaunching [%s] preprocessing=%s encoder=%s decoder=%s:\n%s\n' \
        "$dataset" "$preprocessing" "$encoder" "$decoder" "$cmd"

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

# Format: "task encoder decoder criterion time_limit [extra_args...]"
# All datasets use vit_scratch_p8 — MLP fusion does not use the MT encoder variant.
declare -A datasets=(
    ["hlsburnscars"]="segmentation     vit_scratch_p8  seg_mlp_fusion      cross_entropy  3:00:00"
    ["sen1floods11"]="segmentation     vit_scratch_p8  seg_mlp_fusion      cross_entropy  3:00:00"
    ["mados"]="segmentation           vit_scratch_p8  seg_mlp_fusion      cross_entropy  4:00:00"
    ["pastis"]="segmentation          vit_scratch_p8  seg_mlp_fusion      cross_entropy  6:00:00 task.evaluator.inference_mode=whole"
    ["dynamicen"]="segmentation       vit_scratch_p8  seg_mlp_fusion      cross_entropy  12:00:00"
    ["croptypemapping"]="segmentation  vit_scratch_p8  seg_mlp_fusion      cross_entropy  6:00:00 task.evaluator.inference_mode=whole"
    ["spacenet7cd"]="change_detection  vit_scratch_p8  seg_siam_mlp_fusion dice           4:00:00"
    ["ai4smallfarms"]="segmentation    vit_scratch_p8  seg_mlp_fusion      weighted_bce   6:00:00 task.trainer.best_metric_key=IoU data_replicate=2"
    ["fivebillionpixels"]="segmentation vit_scratch_p8 seg_mlp_fusion      cross_entropy  6:00:00"
)

RUN_DATASETS="all"

for dataset in "${!datasets[@]}"; do
    if [ "$RUN_DATASETS" != "all" ]; then
        [[ " $RUN_DATASETS " != *" $dataset "* ]] && continue
    fi
    read -r task encoder decoder criterion time_limit remainder <<< "${datasets[$dataset]}"
    read -ra extra_args <<< "$remainder"
    launch_job "$dataset" "$task" "$encoder" "$decoder" "$criterion" "$time_limit" "${extra_args[@]}"
done

echo "All jobs queued."
