#!/bin/bash

# -----------------------------------------------------------------------------
# UNIVERSAT MLP-FUSION TEST — HLS Burn Scars
# Same setup as the earlier UPerNet run on this dataset (encoder resolution
# overrides, criterion) -- only the decoder differs, same as the
# ai4smallfarms mlp_fusion test. output_layers stays at the universat.yaml
# default [3, 5, 7, 11] since MLPFusionDecoder uses all of them.
# -----------------------------------------------------------------------------

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/universat_seg"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets/hlsburnscars"
readonly WORK_DIR="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/pangaea_full_seg/mlp_fusion/universat_mlp_fusion"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

sbatch_script="/tmp/universat_hlsburnscars_mlp_fusion_test.sh"

cat > "$sbatch_script" << EOF
#!/bin/bash
#SBATCH --job-name=universat_hlsburnscars_mlp_test
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --clusters=hpda2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=3:00:00
#SBATCH --output=${LOG_DIR}/hlsburnscars_mlp_fusion_output.%j
#SBATCH --error=${LOG_DIR}/hlsburnscars_mlp_fusion_error.%j
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
  decoder=seg_mlp_fusion \\
  preprocessing=seg_default \\
  criterion=cross_entropy \\
  task=segmentation \\
  work_dir=${WORK_DIR} \\
  use_wandb=True
EOF

chmod +x "$sbatch_script"
sbatch "$sbatch_script"
rm "$sbatch_script"
