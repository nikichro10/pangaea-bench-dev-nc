#!/bin/bash

# -----------------------------------------------------------------------------
# SLURM CONFIGURATION
# -----------------------------------------------------------------------------
LOG_DIR="/dss/dsshome1/08/di35pap/projects/pangaea-bench/scripts/logs"
DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
WORK_DIR="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0020/pangaea_experiments"
ENCODER_IMSIZE=128 #s2: 384, l8: 128, remoteclip: 224, prithvi: 224
ENCODER_CONFIG="l8_ijepa" #s2_dino, s2_mae, s2_dinov2, s2_ijepa, l8_dinov2, l8_mae, l8_ijepa
LIMITED_LABEL_TRAIN=0.1

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR/${ENCODER_CONFIG}"
mkdir -p "$WORK_DIR/${ENCODER_CONFIG}"

# Function to determine preprocessing based on size
choose_preprocessing() {
    local dataset_size=$1
    if [ "$dataset_size" -le "$ENCODER_IMSIZE" ]; then
        echo "seg_resize_nonorm"
    else
        echo "seg_default_nonorm"
    fi
}

# Function to create and submit sbatch job
launch_job() {
    local dataset=$1
    local task=$2
    local encoder=$3
    local decoder=$4
    local preprocessing=$5
    local criterion=$6
    local extra_args="$7"

    local job_name="pangaea_${dataset}_${preprocessing}"
    local output_file="${LOG_DIR}/${ENCODER_CONFIG}/${dataset}_${preprocessing}_output.%j"
    local error_file="${LOG_DIR}/${ENCODER_CONFIG}/${dataset}_${preprocessing}_error.%j"
    local sbatch_script="${dataset}_${preprocessing}.sh"

    # Build the full command string first
    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${encoder}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    cmd+=" dataset.root_path='${DATASET_PATH}/${dataset}'"
    cmd+=" work_dir=${WORK_DIR}/${ENCODER_CONFIG}"
    cmd+=" use_wandb=True"

    if [ -n "$extra_args" ]; then
        cmd+=" ${extra_args}"
    fi

    # Echo the final command
    echo  # newline
    echo "Final torchrun command for $dataset:"
    echo "$cmd"

    # Now generate the sbatch script
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
#SBATCH --time=23:00:00
#SBATCH --exclude=hpdar01c02s02,hpdar01c02s03,hpdar01c03s04,hpdar01c04s03,hpdar01c04s04,hpdar04c01s02,hpdar04c05s01,hpdar04c06s03
#SBATCH --output=${output_file}
#SBATCH --error=${error_file}
#SBATCH --chdir=/dss/dsshome1/08/di35pap/projects/pangaea-bench

export CUDA_VISIBLE_DEVICES=0
module add micromamba
micromamba activate pangaea-bench
HYDRA_FULL_ERROR=1

# Execute the command
$cmd
EOF

    # Submit the job
    chmod +x "$sbatch_script"
    sbatch "$sbatch_script"

    # Clean up the script after submission
    rm "$sbatch_script"
}

# Define datasets and their parameters
declare -A dataset_info=(
    ["hlsburnscars"]="segmentation ${ENCODER_CONFIG} seg_upernet cross_entropy 512 limited_label_train=${LIMITED_LABEL_TRAIN}"
    #["sen1floods11"]="segmentation ${ENCODER_CONFIG} seg_upernet cross_entropy 512 limited_label_train=${LIMITED_LABEL_TRAIN}"
    #["pastis"]="segmentation ${ENCODER_CONFIG} seg_upernet_mt_ltae cross_entropy 128 task.evaluator.inference_mode=whole limited_label_train=${LIMITED_LABEL_TRAIN}"
    ["mados"]="segmentation ${ENCODER_CONFIG} seg_upernet cross_entropy 240 limited_label_train=${LIMITED_LABEL_TRAIN}"
    #["spacenet7cd"]="change_detection ${ENCODER_CONFIG} seg_siamupernet_conc dice 256 limited_label_train=${LIMITED_LABEL_TRAIN}"
    #["ai4smallfarms"]="segmentation ${ENCODER_CONFIG} seg_upernet dice 496 task.trainer.best_metric_key=IoU data_replicate=2 limited_label_train=${LIMITED_LABEL_TRAIN}"
    #["croptypemapping"]="segmentation ${ENCODER_CONFIG} seg_upernet_mt_ltae cross_entropy 64 task.evaluator.inference_mode=whole limited_label_train=${LIMITED_LABEL_TRAIN}"
    #["dynamicen"]="segmentation ${ENCODER_CONFIG} seg_upernet_mt_ltae cross_entropy 1024 limited_label_train=${LIMITED_LABEL_TRAIN}"
)

# Process each dataset
for dataset in "${!dataset_info[@]}"; do
    # Extract configuration values
    IFS=' ' read -r task encoder decoder criterion size extra_args <<< "${dataset_info[$dataset]}"

    # Override preprocessing logic for "mados" dataset
    if [ "$dataset" = "mados" ]; then
        # Custom logic for mados: choose between seg_resize or seg_focus_crop based on size
        if [ "$size" -le "$ENCODER_IMSIZE" ]; then
            preprocessing="seg_resize_nonorm"
        else
            preprocessing="seg_focus_crop_nonorm"
        fi
    else
        # Use the default function for other datasets
        preprocessing=$(choose_preprocessing "$size")
    fi

    # Remove quotes from extra_args if present (they're not needed in this format)
    extra_args="${extra_args//\"/}"

    # Launch the job
    launch_job "$dataset" "$task" "$encoder" "$decoder" "$preprocessing" "$criterion" "$extra_args"
done

echo "All jobs queued."
