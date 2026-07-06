#!/bin/bash

# -----------------------------------------------------------------------------
# UNIVERSAT LINEAR-PROBE TEST — AI4SmallFarms
# Evaluates UniverSat the way its paper does: frozen backbone + a single
# linear head on the last (CA_Sub-enriched) feature map, no UPerNet neck/
# multi-depth pyramid. Same general setup (criterion, epoch budget, the
# Dice-collapse workaround) as the UPerNet run on this dataset -- only the
# decoder (and correspondingly encoder.output_layers, since the linear probe
# only ever reads the last level) differs.
#
# output_grid is still == latent_grid here (not yet decoupled -- that's a
# separate, not-yet-implemented change we're deliberately deferring).
# -----------------------------------------------------------------------------

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/universat_seg"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets/ai4smallfarms"
readonly WORK_DIR="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/pangaea_full_seg/linear_probe/universat_seg"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

sbatch_script="/tmp/universat_ai4smallfarms_linear_probe_test.sh"

cat > "$sbatch_script" << EOF
#!/bin/bash
#SBATCH --job-name=universat_ai4smallfarms_lp_test
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --clusters=hpda2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=2:00:00
#SBATCH --output=${LOG_DIR}/ai4smallfarms_linear_probe_output.%j
#SBATCH --error=${LOG_DIR}/ai4smallfarms_linear_probe_error.%j
#SBATCH --chdir=/dss/dsshome1/08/di54zop/pangaea-bench_dev

export HYDRA_FULL_ERROR=1

module add micromamba
eval "\$(micromamba shell hook --shell bash)"
micromamba activate pangaea-bench

torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py \\
  --config-name=train \\
  dataset=ai4smallfarms \\
  dataset.root_path=${DATASET_PATH} \\
  encoder=universat \\
  encoder.input_res.optical=10 \\
  encoder.patch_size_m=160 \\
  encoder.output_layers=[11] \\
  decoder=seg_linear_probe \\
  preprocessing=seg_default \\
  criterion=weighted_bce \\
  task=segmentation \\
  task.trainer.best_metric_key=IoU \\
  data_replicate=2 \\
  work_dir=${WORK_DIR}
EOF

chmod +x "$sbatch_script"
sbatch "$sbatch_script"
rm "$sbatch_script"
