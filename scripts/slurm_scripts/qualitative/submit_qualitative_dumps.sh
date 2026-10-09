#!/bin/bash
# -----------------------------------------------------------------------------
# Submit one GPU job per (dataset, model) that dumps a fixed test sample
# (input RGB, GT, prediction) to scripts/qualitative/out/<dataset>_<model>.npz.
# Runs come from the main PANGAEA benchmark (pangaea_full_seg/upernet).
# -----------------------------------------------------------------------------

set -euo pipefail

readonly PROJECT=/dss/dsshome1/08/di54zop/pangaea-bench_dev
readonly EXPERIMENTS=/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/pangaea_full_seg/upernet
readonly OUT_DIR="${PROJECT}/scripts/qualitative/out"
readonly LOG_DIR="${PROJECT}/scripts/logs/qualitative"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

DATASETS=(sen1floods11 hlsburnscars mados ai4smallfarms)
# label -> benchmark folder under $EXPERIMENTS
declare -A MODEL_FOLDER=(
    [unet]=unet_seg
    [vit_p16_frozen]=vit_scratch_frozen_encoder_seg
    [panopticon]=panopticon_seg
    [dofa]=dofa_seg
    [carl]=carl_imagewise_seg
)

for ds in "${DATASETS[@]}"; do
    for model in unet vit_p16_frozen panopticon dofa carl; do
        run_dir=$(ls -d "${EXPERIMENTS}/${MODEL_FOLDER[$model]}"/*_${ds}_* 2>/dev/null | head -1)
        if [ -z "${run_dir}" ] || [ ! -f "${run_dir}/checkpoint__best.pth" ]; then
            echo "SKIP ${ds}/${model}: no run with checkpoint__best.pth"
            continue
        fi
        job_name="qual_${ds:0:8}_${model:0:6}"
        sbatch_script="${LOG_DIR}/${job_name}.sh"
        cat > "${sbatch_script}" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --clusters=hpda2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=01:30:00
#SBATCH --exclude=${SLURM_EXCLUDE}
#SBATCH --output=${LOG_DIR}/${ds}_${model}_%j.out
#SBATCH --error=${LOG_DIR}/${ds}_${model}_%j.err
#SBATCH --chdir=${PROJECT}

export HYDRA_FULL_ERROR=1

module add micromamba
eval "\$(micromamba shell hook --shell bash)"
micromamba activate pangaea-bench

python scripts/qualitative/dump_test_sample.py \\
    --run_dir "${run_dir}" \\
    --out "${OUT_DIR}/${ds}_${model}.npz"
EOF
        sbatch "${sbatch_script}"
    done
done
