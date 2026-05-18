#!/bin/bash
#SBATCH --job-name=carl_nopad_debug
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --clusters=hpda2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=1:00:00
#SBATCH --exclude=hpdar04c04s02,hpdar04c02s05
#SBATCH --output=/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/carl_nopad/debug_ai4smallfarms_output.%j
#SBATCH --error=/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/carl_nopad/debug_ai4smallfarms_error.%j
#SBATCH --chdir=/dss/dsshome1/08/di54zop/pangaea-bench_dev

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

module add micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate pangaea-bench

torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py \
    --config-name=train \
    dataset=ai4smallfarms \
    encoder=carl \
    decoder=seg_upernet \
    preprocessing=seg_default_nopad \
    criterion=cross_entropy \
    task=segmentation \
    dataset.root_path='/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets/ai4smallfarms' \
    work_dir=/dss/dsshome1/08/di54zop/pangaea_experiments/carl_nopad \
    use_wandb=False
