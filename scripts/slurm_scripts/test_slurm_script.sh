#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=pangaea_test
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --output=/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/output.%j
#SBATCH --error=/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/error.%j
#SBATCH --time=0:20:00
#SBATCH --partition=hpda2_compute_gpu  # hpda2_compute_gpu, hpda2_testgpu
#SBATCH --clusters=hpda2
#SBATCH --mem=32gb
#SBATCH -D /dss/dsshome1/08/di54zop/pangaea-bench_dev
#SBATCH --exclude=hpdar01c02s02,hpdar01c02s03,hpdar01c03s04,hpdar01c04s03,hpdar01c04s04,hpdar04c01s02,hpdar04c05s01,hpdar04c05s05,hpdar04c06s03

export CUDA_VISIBLE_DEVICES=0 #,1,2,3  # Very important to make the GPUs visible
export HYDRA_FULL_ERROR=1

cd /dss/dsshome1/08/di54zop/pangaea-bench_dev

module add micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate pangaea-bench


DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
WORK_DIR="/dss/dsshome1/08/di54zop/pangaea_experiments"

DATASET="sen1floods11"
srun torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=${DATASET} \
   encoder=dofa \
   decoder=seg_upernet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation \
   dataset.root_path="${DATASET_PATH}/${DATASET}" \
   work_dir=${WORK_DIR} \
   use_wandb=True \
   limited_label_train=1.0 \
   task.trainer.n_epochs=80
