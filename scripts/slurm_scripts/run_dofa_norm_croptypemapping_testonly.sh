#!/bin/bash

# Test-only evaluation for croptypemapping_optical normalization experiment.
# Training completed but test evaluation failed (disk quota). This script runs
# inference only using the saved best checkpoint.

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/dofa_norm_exp"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05"

declare -A ckpt_dirs=(
    ["dofa_norm"]="/dss/dsshome1/08/di54zop/pangaea_experiments/dofa_norm_exp/dofa_norm/20260507_093650_1c8d29_dofa_seg_upernet_mt_ltae_croptypemapping_optical_seg_resize_dofa_norm"
    ["custom_global"]="/dss/dsshome1/08/di54zop/pangaea_experiments/dofa_norm_exp/custom_global/20260507_093650_8e64b7_dofa_seg_upernet_mt_ltae_croptypemapping_optical_seg_resize_customglobalnorm"
    ["custom_bandwise"]="/dss/dsshome1/08/di54zop/pangaea_experiments/dofa_norm_exp/custom_bandwise/20260507_093650_e4c41e_dofa_seg_upernet_mt_ltae_croptypemapping_optical_seg_resize_custombandwisenorm"
)

for norm in dofa_norm custom_global custom_bandwise; do
    ckpt_dir="${ckpt_dirs[$norm]}"
    job_name="dofa_croptypemapping_optical_${norm}_test"
    sbatch_script="/tmp/${job_name}.sh"

    cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" train=False"
    cmd+=" ckpt_dir='${ckpt_dir}'"
    cmd+=" use_wandb=False"

    printf 'Launching test-only [croptypemapping_optical] norm=%s\n' "$norm"

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
#SBATCH --time=2:00:00
#SBATCH --exclude=${SLURM_EXCLUDE}
#SBATCH --output=${LOG_DIR}/croptypemapping_optical_${norm}_test_output.%j
#SBATCH --error=${LOG_DIR}/croptypemapping_optical_${norm}_test_error.%j
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

echo "croptypemapping test-only jobs queued."
