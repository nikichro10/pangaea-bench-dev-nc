#!/bin/bash

# SegConcatProbe experiment with DOFA and CARL.
# Multi-scale probe (1.64M params) between SegLinearProbe (0.008M)
# and UPerNet (39M). Encoder is frozen — only projection/fusion/cls trained.
#
# Datasets: sen1floods11, hlsburnscars, mados, spacenet7 (all non-temporal)
# CARL:  seg_default_nopad (no BandPadding, no NormalizeMeanStd — CARL normalizes internally)
# DOFA:  seg_default       (with NormalizeMeanStd)

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/concat_probe"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly WORK_DIR="/dss/dsshome1/08/di54zop/pangaea_experiments/concat_probe"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s02,hpdar04c01s03"

mkdir -p "$LOG_DIR"

# Format: "dataset  encoder  preprocessing  time  [extra_args]"
declare -A jobs=(
    ["sen1floods11_carl_concprobe"]="sen1floods11  carl  seg_default_nopad  cross_entropy  6:00:00"
    ["sen1floods11_dofa_concprobe"]="sen1floods11  dofa  seg_default        cross_entropy  6:00:00"
    ["hlsburnscars_carl_concprobe"]="hlsburnscars  carl  seg_default_nopad  cross_entropy  6:00:00"
    ["hlsburnscars_dofa_concprobe"]="hlsburnscars  dofa  seg_default        cross_entropy  6:00:00"
    ["mados_carl_concprobe"]="mados          carl  seg_focus_crop_nopad  cross_entropy  8:00:00"
    ["mados_dofa_concprobe"]="mados          dofa  seg_focus_crop        cross_entropy  8:00:00"
    ["spacenet7_carl_concprobe"]="spacenet7     carl  seg_default_nopad  dice           8:00:00"
    ["spacenet7_dofa_concprobe"]="spacenet7     dofa  seg_default        dice           8:00:00"
)

for key in "${!jobs[@]}"; do
    read -r dataset encoder preprocessing criterion time_limit remainder <<< "${jobs[$key]}"
    read -ra extra_args <<< "$remainder"

    job_name="concat_probe_${key}"
    sbatch_script="/tmp/${job_name}.sh"

    cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${encoder}"
    cmd+=" decoder=seg_concat_probe"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=segmentation"
    cmd+=" dataset.root_path='${DATASET_PATH}/${dataset}'"
    cmd+=" work_dir=${WORK_DIR}"
    cmd+=" use_wandb=True"
    cmd+=" task.trainer.ckpt_interval=999"
    cmd+=" finetune=false"
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

echo "All concat_probe jobs queued."
