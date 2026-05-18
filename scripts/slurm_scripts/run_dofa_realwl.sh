#!/bin/bash

# DOFA with sensor-specific (real) wavelengths experiment.
# Tests whether using true central wavelengths improves DOFA's dynamic
# kernel generation compared to S2 defaults.
#
# SpaceNet7: Planet PS2 (Dove) RGB wavelengths via dofa_planet_rgb encoder
#
# Note: FiveBillionPixels/GF-2 already ran as dofa_gf2wl (mIoU=67.635).

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/dofa_realwl"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly WORK_DIR="/dss/dsshome1/08/di54zop/pangaea_experiments/dofa_realwl"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05"

mkdir -p "$LOG_DIR"

job_name="dofa_realwl_spacenet7"
sbatch_script="/tmp/${job_name}.sh"

cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
cmd+=" --config-name=train"
cmd+=" dataset=spacenet7"
cmd+=" encoder=dofa_planet_rgb"
cmd+=" decoder=seg_upernet"
cmd+=" preprocessing=seg_default"
cmd+=" criterion=cross_entropy"
cmd+=" task=segmentation"
cmd+=" dataset.root_path='${DATASET_PATH}/spacenet7'"
cmd+=" work_dir=${WORK_DIR}"
cmd+=" use_wandb=True"
cmd+=" task.trainer.ckpt_interval=999"

printf 'Launching [spacenet7]\n'

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
#SBATCH --time=12:00:00
#SBATCH --exclude=${SLURM_EXCLUDE}
#SBATCH --output=${LOG_DIR}/spacenet7_output.%j
#SBATCH --error=${LOG_DIR}/spacenet7_error.%j
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

echo "All dofa_realwl jobs queued."
