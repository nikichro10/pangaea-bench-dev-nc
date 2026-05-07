#!/bin/bash
#SBATCH --job-name=dofa_hls_dofa_norm
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --clusters=hpda2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=6:00:00
#SBATCH --exclude=hpdar04c04s02,hpdar04c02s05
#SBATCH --output=/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/dofa_seg/hlsburnscars_dofa_norm_output.%j
#SBATCH --error=/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/dofa_seg/hlsburnscars_dofa_norm_error.%j
#SBATCH --chdir=/dss/dsshome1/08/di54zop/pangaea-bench_dev

export HYDRA_FULL_ERROR=1

module add micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate pangaea-bench

torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py \
    --config-name=train \
    dataset=hlsburnscars \
    encoder=dofa \
    decoder=seg_upernet \
    preprocessing=seg_default_dofa_norm \
    criterion=cross_entropy \
    task=segmentation \
    dataset.root_path='/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets/hlsburnscars' \
    work_dir=/dss/dsshome1/08/di54zop/pangaea_experiments/dofa_seg \
    use_wandb=True
