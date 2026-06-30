#!/bin/bash

# UNet Band Subset Degradation (native UNet decoder)
#
# Baseline: UNet trained from scratch with finetune=true.
# Uses seg_unet decoder (native skip-connection decoder matching UNet encoder architecture).
# UPerNet is incompatible with UNet: UNet encoder sets output_layers=None which causes
# a TypeError in SegUPerNet.__init__ (len(None)).
#
# Encoder/decoder pairs:
#   - MADOS / Sen1Floods11 (single-temporal): unet_encoder + seg_unet + seg_resize
#   - PASTIS (multi-temporal, T=6):           unet_encoder_mi + seg_unet + seg_default
#     (single-temporal unet_encoder cannot handle T=6; squeeze(2) is a no-op when T!=1)
#
# Matches the existing pangaea_full_seg UNet baseline runs.

readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/band_degradation/unet"
readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly WORK_DIR="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/band_subset_degradation/upernet/unet_band_degradation"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

# Snapshot of active SLURM jobs fetched once — used by launch_job to skip duplicates
_ACTIVE_JOB_NAMES=$(squeue --clusters=hpda2 -u "$USER" -h --format="%j" 2>/dev/null | tr '\n' '|')
_ACTIVE_JOB_IDS=$(squeue --clusters=hpda2 -u "$USER" -h --format="%i" 2>/dev/null | tr '\n' '|')

# launch_job <dataset> <band_level> <task> <decoder> <criterion> <time_limit>
#            <preprocessing> <encoder_name> [extra hydra args...]
launch_job() {
    local -r dataset=$1
    local -r band_level=$2
    local -r task=$3
    local -r decoder=$4
    local -r criterion=$5
    local -r time_limit=$6
    local -r preprocessing=$7
    local -r encoder_name=$8
    local -ra extra_args=("${@:9}")

    local -r job_name="unet_${dataset}_${band_level}"
    local -r sbatch_script="${job_name}.sh"

    # Skip if this exact job is already pending/running
    if echo "|${_ACTIVE_JOB_NAMES}|" | grep -qF "|${job_name}|"; then
        printf 'SKIP (queued)  [%s | %s]\n' "$dataset" "$band_level"
        return 0
    fi
    # Skip if a completed (non-active) run already exists for this dataset+band config
    local _n_bands _ds _d _cfg _jid _dn _nb
    _n_bands=$(echo "${extra_args[-1]:-}" | grep -oE 'B[0-9]+[A-Za-z]*' | wc -l | tr -d ' ')
    case "$dataset" in mados) _ds="MADOS";; pastis) _ds="Pastis";; sen1floods11) _ds="Sen1Floods11";; esac
    for _d in "${WORK_DIR}"/*/; do
        [ -f "${_d}checkpoint__best.pth" ] || continue
        _cfg="${_d}configs/config.yaml"; [ -f "$_cfg" ] || continue
        _jid=$(basename "$_d" | cut -d_ -f2)
        echo "|${_ACTIVE_JOB_IDS}|" | grep -qF "|${_jid}|" && continue
        _dn=$(awk '/^  dataset_name:/{print $2; exit}' "$_cfg")
        _nb=$(grep -A 50 "input_bands:" "$_cfg" | grep -c "^    - B" || true)
        [ "$_dn" = "$_ds" ] && [ "$_nb" = "$_n_bands" ] && {
            printf 'SKIP (done)    [%s | %s]\n' "$dataset" "$band_level"
            return 0
        }
    done

    trap 'rm -f "${sbatch_script}"' EXIT

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${dataset}"
    cmd+=" encoder=${encoder_name}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    cmd+=" finetune=true"
    cmd+=" dataset.root_path='${DATASET_PATH}/${dataset}'"
    cmd+=" work_dir=${WORK_DIR}"
    cmd+=" use_wandb=True"
    cmd+=" wandb_project=geofm-bench-band-degradation"

    if [ "${#extra_args[@]}" -gt 0 ]; then
        cmd+=" ${extra_args[*]}"
    fi

    printf '\nLaunching [%s | %s] encoder=%s preprocessing=%s:\n%s\n' \
        "$dataset" "$band_level" "$encoder_name" "$preprocessing" "$cmd"

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
#SBATCH --time=${time_limit}
#SBATCH --exclude=${SLURM_EXCLUDE}
#SBATCH --output=${LOG_DIR}/${dataset}_${band_level}_output.%j
#SBATCH --error=${LOG_DIR}/${dataset}_${band_level}_error.%j
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
    trap - EXIT
}

declare -A mados_bands=(
    ["rgb"]="'encoder.input_bands={optical:[B2,B3,B4]}'"
    ["rgb_nir"]="'encoder.input_bands={optical:[B2,B3,B4,B8]}'"
    ["rgb_rededge"]="'encoder.input_bands={optical:[B2,B3,B4,B5,B6,B7,B8,B8A]}'"
    ["rgb_swir"]="'encoder.input_bands={optical:[B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12]}'"
    ["full"]="'encoder.input_bands={optical:[B1,B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12]}'"
)

declare -A pastis_bands=(
    ["rgb"]="'encoder.input_bands={optical:[B2,B3,B4]}'"
    ["rgb_nir"]="'encoder.input_bands={optical:[B2,B3,B4,B8]}'"
    ["rgb_rededge"]="'encoder.input_bands={optical:[B2,B3,B4,B5,B6,B7,B8,B8A]}'"
    ["full"]="'encoder.input_bands={optical:[B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12]}'"
)

declare -A sen1floods11_bands=(
    ["rgb"]="'encoder.input_bands={optical:[B2,B3,B4]}'"
    ["rgb_nir"]="'encoder.input_bands={optical:[B2,B3,B4,B8]}'"
    ["rgb_rededge"]="'encoder.input_bands={optical:[B2,B3,B4,B5,B6,B7,B8,B8A]}'"
    ["rgb_swir"]="'encoder.input_bands={optical:[B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12]}'"
    ["full"]="'encoder.input_bands={optical:[B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12]}'"
)

RUN_LEVELS="all"

for band_level in "${!mados_bands[@]}"; do
    if [ "$RUN_LEVELS" != "all" ] && [[ " $RUN_LEVELS " != *" $band_level "* ]]; then continue; fi
    launch_job mados "$band_level" segmentation seg_unet cross_entropy 8:00:00 \
        seg_resize unet_encoder "${mados_bands[$band_level]}"
done

for band_level in "${!pastis_bands[@]}"; do
    if [ "$RUN_LEVELS" != "all" ] && [[ " $RUN_LEVELS " != *" $band_level "* ]]; then continue; fi
    launch_job pastis "$band_level" segmentation seg_unet cross_entropy 12:00:00 \
        seg_default unet_encoder_mi "${pastis_bands[$band_level]}"
done

for band_level in "${!sen1floods11_bands[@]}"; do
    if [ "$RUN_LEVELS" != "all" ] && [[ " $RUN_LEVELS " != *" $band_level "* ]]; then continue; fi
    launch_job sen1floods11 "$band_level" segmentation seg_unet cross_entropy 6:00:00 \
        seg_resize unet_encoder "${sen1floods11_bands[$band_level]}"
done

echo "All jobs queued."
