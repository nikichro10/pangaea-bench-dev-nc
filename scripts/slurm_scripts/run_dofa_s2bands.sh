#!/bin/bash

# DOFA with fixed S2 input_bands + BandPadding experiment.
# Tests whether providing the full S2 wavelength context (with zero-padded
# missing bands) improves DOFA's dynamic kernel generation compared to
# using only the dataset's available bands.
#
# encoder=dofa_s2bands fixes input_bands to all 13 S2 optical bands.
# BandPadding in the preprocessing zero-fills missing bands so the
# hypernetwork always generates 13 wavelength-specific kernels.
# NormalizeMeanStd runs before BandPadding so zeros stay as zeros.

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/dofa_s2bands"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly WORK_DIR="/dss/dsshome1/08/di54zop/pangaea_experiments/dofa_s2bands"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s02,hpdar04c01s03"

mkdir -p "$LOG_DIR"

# Format: "dataset  decoder  preprocessing  criterion  time  [extra_args]"
declare -A jobs=(
    ["fivebillionpixels"]="fivebillionpixels   seg_upernet          seg_default_pad    cross_entropy  24:00:00"
    ["hlsburnscars"]="hlsburnscars             seg_upernet          seg_default_pad    cross_entropy  8:00:00"
    ["sen1floods11"]="sen1floods11             seg_upernet          seg_default_pad    cross_entropy  8:00:00"
    ["mados"]="mados                          seg_upernet          seg_focus_crop_pad cross_entropy  8:00:00"
    ["ai4smallfarms"]="ai4smallfarms           seg_upernet          seg_default_pad    dice           24:00:00 task.trainer.best_metric_key=IoU data_replicate=2"
    ["dynamicen"]="dynamicen                  seg_upernet_mt_ltae  seg_default_pad    cross_entropy  12:00:00"
    ["pastis_optical"]="pastis_optical         seg_upernet_mt_ltae  seg_resize_pad     cross_entropy  12:00:00 task.evaluator.inference_mode=whole"
    ["croptypemapping_optical"]="croptypemapping_optical seg_upernet_mt_ltae seg_resize_pad cross_entropy 12:00:00 task.evaluator.inference_mode=whole"
    ["spacenet7"]="spacenet7                  seg_upernet          seg_default_pad    dice           12:00:00"
)

for key in "${!jobs[@]}"; do
    read -r dataset decoder preprocessing criterion time_limit remainder <<< "${jobs[$key]}"
    read -ra extra_args <<< "$remainder"

    base_dataset="${dataset%_optical}"
    job_name="dofa_s2bands_${key}"
    sbatch_script="/tmp/${job_name}.sh"

    cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=dofa_s2bands"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=segmentation"
    cmd+=" dataset.root_path='${DATASET_PATH}/${base_dataset}'"
    cmd+=" work_dir=${WORK_DIR}"
    cmd+=" use_wandb=True"
    cmd+=" task.trainer.ckpt_interval=999"
    [ "${#extra_args[@]}" -gt 0 ] && cmd+=" ${extra_args[*]}"

    printf 'Launching [%s]\n' "$key"

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
#SBATCH --output=${LOG_DIR}/${key}_output.%j
#SBATCH --error=${LOG_DIR}/${key}_error.%j
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
done

echo "All dofa_s2bands jobs queued."
