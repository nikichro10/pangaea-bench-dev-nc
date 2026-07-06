#!/bin/bash

# -----------------------------------------------------------------------------
# UNIVERSAT MLP-FUSION TEST — AI4SmallFarms
# Same setup as the earlier UPerNet run on this dataset (encoder overrides,
# criterion, Dice-collapse workaround) -- only the decoder differs. Unlike
# the linear-probe decoder, MLPFusionDecoder uses all of encoder.output_layers
# (n_layers = len(encoder.output_layers)), so output_layers stays at the
# universat.yaml default [3, 5, 7, 11] here, not overridden to a single level.
# -----------------------------------------------------------------------------

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/universat_seg"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets/ai4smallfarms"
readonly WORK_DIR="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/pangaea_full_seg/mlp_fusion/universat_mlp_fusion"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

sbatch_script="/tmp/universat_ai4smallfarms_mlp_fusion_test.sh"

cat > "$sbatch_script" << EOF
#!/bin/bash
#SBATCH --job-name=universat_ai4smallfarms_mlp_test
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --clusters=hpda2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=6:00:00
#SBATCH --output=${LOG_DIR}/ai4smallfarms_mlp_fusion_output.%j
#SBATCH --error=${LOG_DIR}/ai4smallfarms_mlp_fusion_error.%j
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
  decoder=seg_mlp_fusion \\
  preprocessing=seg_default \\
  criterion=weighted_bce \\
  task=segmentation \\
  task.trainer.best_metric_key=IoU \\
  data_replicate=2 \\
  work_dir=${WORK_DIR} \\
  use_wandb=True
EOF

chmod +x "$sbatch_script"
sbatch "$sbatch_script"
rm "$sbatch_script"
