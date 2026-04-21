#!/bin/bash

# -----------------------------------------------------------------------------
# SLURM CONFIGURATION
# -----------------------------------------------------------------------------
readonly LOG_DIR="/dss/dsshome1/08/di35pap/projects/pangaea-bench/scripts/logs"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly WORK_DIR="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0020/pangaea_experiments"

# -----------------------------------------------------------------------------
# ENCODER CONFIGURATION
# -----------------------------------------------------------------------------
# Set the encoder input image size (ignored for U-Net baseline), necessary to pick between resize and default
# TODO: Different imsizes for multiple canonical sensors (e.g. S2, L8)
readonly ENCODER_IMSIZE=384  # s2: 384, l8: 128, remoteclip: 224, prithvi: 224, terramind_base: 224
readonly ENCODER_CONFIG="hierab_mm_jepa_lowlambdadetach.yaml" # Set to unet_encoder or a foundation model config name
readonly LIMITED_LABEL_TRAIN=1.0  # run this script again for: 1.0, 0.5 and 0.1 if required
readonly FINETUNE=false  # Set to true to add finetune=true to the command (default in Pangaea: false)

readonly SLURM_EXCLUDE_NODES="hpdar01c02s02,hpdar01c02s03,hpdar01c03s04,hpdar01c04s03,hpdar01c04s04,hpdar04c01s02,hpdar04c05s01,hpdar04c05s05,hpdar04c06s03"

# Create required directories if they don't exist
mkdir -p "${LOG_DIR}/${ENCODER_CONFIG}"
mkdir -p "${WORK_DIR}/${ENCODER_CONFIG}"

# Determine preprocessing mode based on encoder, dataset name, and image size.
# unet_encoder and unet_encoder_mi always use seg_default (native image size of each dataset, no resizing needed).
# For other encoders: mados uses seg_focus_crop when size exceeds ENCODER_IMSIZE;
# all other datasets use seg_default in that case.
choose_preprocessing() {
    local -r dataset=$1
    local -r dataset_size=$2

    if [ "$ENCODER_CONFIG" = "unet_encoder" ] || [ "$ENCODER_CONFIG" = "unet_encoder_mi" ]; then
        echo "seg_default"
        return
    fi

    if [ "$dataset_size" -le "$ENCODER_IMSIZE" ]; then
        echo "seg_resize"
    elif [ "$dataset" = "mados" ]; then
        echo "seg_focus_crop"
    else
        echo "seg_default"
    fi
}

# Create a temporary sbatch script, submit it, and remove it afterwards.
launch_job() {
    local -r dataset=$1
    local -r task=$2
    local -r encoder=$3
    local -r decoder=$4
    local -r preprocessing=$5
    local -r criterion=$6
    # Remaining arguments are treated as an array of extra hydra overrides.
    local -ra extra_args=("${@:7}")

    local -r job_name="pangaea_${dataset}_${preprocessing}"
    local -r output_file="${LOG_DIR}/${ENCODER_CONFIG}/${dataset}_${preprocessing}_output.%j"
    local -r error_file="${LOG_DIR}/${ENCODER_CONFIG}/${dataset}_${preprocessing}_error.%j"
    local -r sbatch_script="${dataset}_${preprocessing}.sh"

    # Ensure the temporary sbatch script is removed even if the script exits early.
    trap 'rm -f "${sbatch_script}"' EXIT

    # Build the torchrun command
    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${encoder}"

    # unet_encoder and unet_encoder_mi always use seg_unet as decoder and require finetune=true.
    #cmd+=" decoder=seg_unet"
    if [ "$ENCODER_CONFIG" = "unet_encoder" ] || [ "$ENCODER_CONFIG" = "unet_encoder_mi" ]; then
        cmd+=" decoder=seg_siamunet_diff"
        cmd+=" finetune=true"
    else
        cmd+=" decoder=${decoder}"
    fi

    if [ "$FINETUNE" = true ]; then
        cmd+=" finetune=true"
    fi

    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    cmd+=" dataset.root_path='${DATASET_PATH}/${dataset}'"
    cmd+=" work_dir=${WORK_DIR}/${ENCODER_CONFIG}"
    cmd+=" use_wandb=True"

    if [ "${#extra_args[@]}" -gt 0 ]; then
        cmd+=" ${extra_args[*]}"
    fi

    printf '\nFinal torchrun command for %s:\n%s\n' "$dataset" "$cmd"

    # Generate the sbatch script
    cat > "$sbatch_script" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --clusters=hpda2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --exclude=${SLURM_EXCLUDE_NODES}
#SBATCH --output=${output_file}
#SBATCH --error=${error_file}
#SBATCH --chdir=/dss/dsshome1/08/di35pap/projects/pangaea-bench

export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
module add micromamba
micromamba activate pangaea-bench

# Execute the command
$cmd
EOF

    chmod +x "$sbatch_script"
    sbatch "$sbatch_script"
    rm "$sbatch_script"

    # Defuse the EXIT trap now that cleanup is done.
    trap - EXIT
}

# Define datasets and their parameters.
# Format: "task encoder decoder criterion size [extra_args...]"
declare -A dataset_info=(
    ["hlsburnscars"]="segmentation ${ENCODER_CONFIG} seg_upernet cross_entropy 512 limited_label_train=${LIMITED_LABEL_TRAIN}"
    ["sen1floods11"]="segmentation ${ENCODER_CONFIG} seg_upernet cross_entropy 512 limited_label_train=${LIMITED_LABEL_TRAIN}"
    ["pastis"]="segmentation ${ENCODER_CONFIG} seg_upernet_mt_ltae cross_entropy 128 task.evaluator.inference_mode=whole limited_label_train=${LIMITED_LABEL_TRAIN}"
    ["mados"]="segmentation ${ENCODER_CONFIG} seg_upernet cross_entropy 240 limited_label_train=${LIMITED_LABEL_TRAIN}"
    ["spacenet7cd"]="change_detection ${ENCODER_CONFIG} seg_siamupernet_conc dice 256 limited_label_train=${LIMITED_LABEL_TRAIN}"
    ["ai4smallfarms"]="segmentation ${ENCODER_CONFIG} seg_upernet dice 496 task.trainer.best_metric_key=IoU data_replicate=2 limited_label_train=${LIMITED_LABEL_TRAIN}"
    #["croptypemapping"]="segmentation ${ENCODER_CONFIG} seg_upernet_mt_ltae cross_entropy 64 task.evaluator.inference_mode=whole limited_label_train=${LIMITED_LABEL_TRAIN}"
    ["dynamicen"]="segmentation ${ENCODER_CONFIG} seg_upernet_mt_ltae cross_entropy 1024 limited_label_train=${LIMITED_LABEL_TRAIN}"
)

# Process each dataset
for dataset in "${!dataset_info[@]}"; do
    # Read the five fixed fields; the rest of the line goes into 'remainder'.
    read -r task encoder decoder criterion size remainder <<< "${dataset_info[$dataset]}"
    read -ra extra_args <<< "$remainder"

    preprocessing=$(choose_preprocessing "$dataset" "$size")

    launch_job "$dataset" "$task" "$encoder" "$decoder" "$preprocessing" "$criterion" "${extra_args[@]}"
done

echo "All jobs queued."
