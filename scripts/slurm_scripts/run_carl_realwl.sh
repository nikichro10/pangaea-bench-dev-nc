#!/bin/bash

# CARL with sensor-specific (real) wavelengths experiment.
# Tests whether using the true central wavelengths of each sensor
# (instead of S2 defaults) improves CARL's spectral encoding.
#
# FiveBillionPixels: Gaofen-2 (GF-2) wavelengths via carl_gf2 encoder
# SpaceNet7:         Planet PS2 (Dove) RGB wavelengths via carl_planet_rgb encoder
#
# Both runs use nopad preprocessing (no BandPadding, no NormalizeMeanStd)
# so set_wv_list_from_data_bands() restricts wv_list to actual dataset bands.

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/carl_realwl"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly WORK_DIR="/dss/dsshome1/08/di54zop/pangaea_experiments/carl_realwl"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05"

mkdir -p "$LOG_DIR"

# Format: "dataset  encoder  decoder  preprocessing  time"
declare -A jobs=(
    ["fivebillionpixels"]="fivebillionpixels carl_gf2         seg_upernet seg_default_nopad 24:00:00"
    ["spacenet7"]="spacenet7         carl_planet_rgb  seg_upernet seg_default_nopad 12:00:00"
)

for key in "${!jobs[@]}"; do
    read -r dataset encoder decoder preprocessing time_limit <<< "${jobs[$key]}"

    job_name="carl_realwl_${key}"
    sbatch_script="/tmp/${job_name}.sh"

    cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${encoder}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=cross_entropy"
    cmd+=" task=segmentation"
    cmd+=" dataset.root_path='${DATASET_PATH}/${dataset}'"
    cmd+=" work_dir=${WORK_DIR}"
    cmd+=" use_wandb=True"
    cmd+=" task.trainer.ckpt_interval=999"

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

echo "All carl_realwl jobs queued."
