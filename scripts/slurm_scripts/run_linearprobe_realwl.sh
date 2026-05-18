#!/bin/bash

# Linear probe experiment: real vs. S2 wavelengths for CARL and DOFA.
# Goal: test whether true sensor wavelengths change encoder representations,
# using a frozen linear head so only representation quality is measured.
#
# FiveBillionPixels: S2 defaults vs. Gaofen-2 (GF-2) wavelengths
# SpaceNet7:         S2 defaults vs. Planet PS2 (Dove) wavelengths
#
# SpaceNet7 requires dataset.num_classes=2 override (dataset config says 1
# but has Background=0 and Building=1 as targets).

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/linearprobe_realwl"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly WORK_DIR="/dss/dsshome1/08/di54zop/pangaea_experiments/linearprobe_realwl"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05"

mkdir -p "$LOG_DIR"

# Format: "dataset  encoder  preprocessing  extra_args  time"
declare -A jobs=(
    ["fbp_carl_s2"]="fivebillionpixels   carl             seg_default_nopad  ''                      8:00:00"
    ["fbp_carl_gf2"]="fivebillionpixels  carl_gf2         seg_default_nopad  ''                      8:00:00"
    ["fbp_dofa_s2"]="fivebillionpixels   dofa             seg_default        ''                      8:00:00"
    ["fbp_dofa_gf2"]="fivebillionpixels  dofa_gf2wl       seg_default        ''                      8:00:00"
    ["sn7_carl_s2"]="spacenet7           carl             seg_default_nopad  'dataset.num_classes=2' 4:00:00"
    ["sn7_carl_planet"]="spacenet7        carl_planet_rgb  seg_default_nopad  'dataset.num_classes=2' 4:00:00"
    ["sn7_dofa_s2"]="spacenet7           dofa             seg_default        'dataset.num_classes=2' 4:00:00"
    ["sn7_dofa_planet"]="spacenet7        dofa_planet_rgb  seg_default        'dataset.num_classes=2' 4:00:00"
)

for key in "${!jobs[@]}"; do
    read -r dataset encoder preprocessing extra time_limit <<< "${jobs[$key]}"

    # Strip surrounding quotes from extra
    extra="${extra//\'/}"

    base_dataset="${dataset}"
    job_name="lp_realwl_${key}"
    sbatch_script="/tmp/${job_name}.sh"

    cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${encoder}"
    cmd+=" decoder=seg_linear_probe"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=cross_entropy"
    cmd+=" task=segmentation"
    cmd+=" dataset.root_path='${DATASET_PATH}/${base_dataset}'"
    cmd+=" work_dir=${WORK_DIR}"
    cmd+=" use_wandb=True"
    cmd+=" task.trainer.ckpt_interval=999"
    cmd+=" finetune=false"
    [ -n "$extra" ] && cmd+=" ${extra}"

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

echo "All linearprobe_realwl jobs queued."
