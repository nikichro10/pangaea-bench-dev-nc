#!/bin/bash
#SBATCH --job-name=panopticon_hlsburnscars
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --clusters=hpda2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=1:00:00
#SBATCH --exclude=hpdar04c04s02,hpdar04c02s05,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c06s02
#SBATCH --output=/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/panopticon_test/hlsburnscars_output.%j
#SBATCH --error=/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/panopticon_test/hlsburnscars_error.%j
#SBATCH --chdir=/dss/dsshome1/08/di54zop/pangaea-bench_dev

export HYDRA_FULL_ERROR=1

module add micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate pangaea-bench

torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py \
  --config-name=train \
  dataset=hlsburnscars \
  encoder=panopticon \
  decoder=seg_upernet \
  preprocessing=seg_default \
  criterion=cross_entropy \
  task=segmentation \
  dataset.root_path=/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets/hlsburnscars \
  work_dir=/dss/dsshome1/08/di54zop/pangaea_experiments/panopticon_test \
  task.trainer.n_epochs=5 \
  task.trainer.eval_interval=1 \
  task.trainer.ckpt_interval=5
