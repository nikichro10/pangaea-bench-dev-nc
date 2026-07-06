#!/bin/bash

# -----------------------------------------------------------------------------
# UNIVERSAT SMOKE TEST — HLS Burn Scars
# First real end-to-end run of the new UniverSat encoder (universat_encoder.py):
# frozen backbone + UPerNet decoder. Scoped down to 2 epochs purely to verify
# the pipeline (data loading, preprocessing, encoder forward, decoder,
# eval) runs cleanly on a GPU node — not a real benchmark run.
#
# HLS is Harmonized Landsat-Sentinel, native 30m/pixel (not the 10m Sentinel-2
# default baked into configs/encoder/universat.yaml), hence the input_res and
# patch_size_m overrides below: patch_size_m=480 at 30m/pixel gives the same
# 16px-equivalent patch grid (224/16=14 train, 512/16=32 full-image eval) that
# the default 160m/10m combination gives for Sentinel-2 data.
# -----------------------------------------------------------------------------

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/universat_test"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets/hlsburnscars"
readonly WORK_DIR="/dss/dsshome1/08/di54zop/pangaea_experiments/universat_test_hlsburnscars"

mkdir -p "${LOG_DIR}"

sbatch_script="/tmp/universat_hlsburnscars_test.sh"

cat > "$sbatch_script" << EOF
#!/bin/bash
#SBATCH --job-name=universat_hlsburnscars_test
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --clusters=hpda2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=1:00:00
#SBATCH --output=${LOG_DIR}/output.%j
#SBATCH --error=${LOG_DIR}/error.%j
#SBATCH --chdir=/dss/dsshome1/08/di54zop/pangaea-bench_dev

export HYDRA_FULL_ERROR=1

module add micromamba
eval "\$(micromamba shell hook --shell bash)"
micromamba activate pangaea-bench

torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py \\
  --config-name=train \\
  dataset=hlsburnscars \\
  dataset.root_path=${DATASET_PATH} \\
  encoder=universat \\
  encoder.input_res.optical=30 \\
  encoder.patch_size_m=480 \\
  decoder=seg_upernet \\
  preprocessing=seg_default \\
  criterion=cross_entropy \\
  task=segmentation \\
  task.trainer.n_epochs=2 \\
  task.trainer.eval_interval=1 \\
  task.trainer.ckpt_interval=1 \\
  work_dir=${WORK_DIR}
EOF

chmod +x "$sbatch_script"
sbatch "$sbatch_script"
rm "$sbatch_script"
