#!/bin/bash
#
# P1 -- Minimal (nearly-degenerate neighbor swaps) wavelength-label ablation.
#
# Tests whether wavelength-conditioned encoders (DOFA, DOFAv2, Panopticon,
# Copernicus-FM, CARL, SMARTIES) generate physically sensible weights through
# their hypernetwork / dynamic-projection mechanism. Pixel data and per-band
# normalization stay completely untouched at their original channel position;
# only the *wavelength label* fed to the encoder's weight generator is swapped
# between two bands whose true centre wavelengths are nearly identical. A
# physically-grounded model should barely notice -- this is the sanity floor
# of the wavelength-swap ladder. Because data and normalization are never
# touched, any accuracy drop here can only be attributed to the spectral
# conditioning mechanism itself.
#
# Swapped pairs (values are exchanged in place from each encoder's own
# existing wave_list / proj_idx table -- no external "true" wavelength is
# injected, just the assignment of the pair is swapped):
#
#   S2 11/12-band datasets (mados, m-cashew-plantation, m-SA-crop-type):
#     B8  <-> B8A   (842/865 nm,   Delta 23 nm)
#     B4  <-> B5    (665/705 nm,   Delta 40 nm)
#     B6  <-> B7    (740/783 nm,   Delta 43 nm)
#   6-band dataset (m-ForestNet, S2-aliased Landsat-8 bands):
#     B11 <-> B12   (SWIR1/SWIR2, 1610/2190 nm)
#
# Per-encoder implementation:
#   DOFA / DOFAv2 / Panopticon / Copernicus-FM -- plain CLI override on the
#     existing encoder.wave_list.optical.{band} keys (no '+' needed, keys
#     already exist in the base config).
#   CARL          -- encoder.band_wavelength_overrides.optical.{band} (µm),
#     an override dict purpose-built for permutation ablations; new leaf
#     keys require '+'.
#   SMARTIES      -- doesn't use continuous wavelength, only a discrete
#     projection-head index per band. The analogous swap exchanges
#     encoder.proj_idx_override.optical.{band} indices instead.
#
# All encoders frozen (finetune=false), matching the existing true-wl/const-wl
# wavelength ablations, to isolate the effect on the pretrained representation.
#
# W&B project: wavelength-ablation (new, dedicated to this experiment).
#
# 6 encoders x 4 datasets = 24 runs.
#
# Override RUN_ENCODERS / RUN_DATASETS at call time for partial runs:
#   RUN_ENCODERS="dofa carl_imagewise" RUN_DATASETS="mados" bash run_p1_wlswap_ablation.sh

set -euo pipefail

readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly GEO_BENCH_DATA="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/datasets/geobench"
readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/p1_wlswap_ablation"
readonly WORK_BASE="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/p1_wlswap_ablation"
readonly WANDB_PROJECT="wavelength-ablation"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}" "${WORK_BASE}"

_ACTIVE_JOB_NAMES=$(squeue --clusters=hpda2 -u "$USER" -h --format="%j" 2>/dev/null | tr '\n' '|')
job_in_queue() { echo "|${_ACTIVE_JOB_NAMES}|" | grep -qF "|${1}|"; }

# -----------------------------------------------------------------------------
# swap_args <enc_key> <pairset: s2|l8>
# Echoes the Hydra CLI overrides that swap the wavelength/proj_idx assignment
# of the given band pair(s) for one encoder.
# -----------------------------------------------------------------------------
swap_args() {
    local -r enc=$1 pairset=$2
    local args=""

    case "$enc" in
        dofa|dofav2)  # micrometres, keys already exist in base config
            if [ "$pairset" = "s2" ]; then
                args="encoder.wave_list.optical.B8=0.864"
                args+=" encoder.wave_list.optical.B8A=0.832"
                args+=" encoder.wave_list.optical.B4=0.705"
                args+=" encoder.wave_list.optical.B5=0.665"
                args+=" encoder.wave_list.optical.B6=0.783"
                args+=" encoder.wave_list.optical.B7=0.74"
            else
                args="encoder.wave_list.optical.B11=2.20"
                args+=" encoder.wave_list.optical.B12=1.61"
            fi
            ;;
        panopticon)  # nanometres, keys already exist in base config
            if [ "$pairset" = "s2" ]; then
                args="encoder.wave_list.optical.B8=864.780"
                args+=" encoder.wave_list.optical.B8A=827.539"
                args+=" encoder.wave_list.optical.B4=704.006"
                args+=" encoder.wave_list.optical.B5=664.630"
                args+=" encoder.wave_list.optical.B6=782.419"
                args+=" encoder.wave_list.optical.B7=740.552"
            else
                args="encoder.wave_list.optical.B11=2203.618"
                args+=" encoder.wave_list.optical.B12=1613.862"
            fi
            ;;
        copernicus_fm_base)  # nanometres, keys already exist in base config
            if [ "$pairset" = "s2" ]; then
                args="encoder.wave_list.optical.B8=865"
                args+=" encoder.wave_list.optical.B8A=842"
                args+=" encoder.wave_list.optical.B4=705"
                args+=" encoder.wave_list.optical.B5=665"
                args+=" encoder.wave_list.optical.B6=783"
                args+=" encoder.wave_list.optical.B7=740"
            else
                args="encoder.wave_list.optical.B11=2190"
                args+=" encoder.wave_list.optical.B12=1610"
            fi
            ;;
        carl_imagewise)  # micrometres, override dict -- new leaf keys need '+'
            if [ "$pairset" = "s2" ]; then
                args="+encoder.band_wavelength_overrides.optical.B8=0.865"
                args+=" +encoder.band_wavelength_overrides.optical.B8A=0.842"
                args+=" +encoder.band_wavelength_overrides.optical.B4=0.705"
                args+=" +encoder.band_wavelength_overrides.optical.B5=0.665"
                args+=" +encoder.band_wavelength_overrides.optical.B6=0.783"
                args+=" +encoder.band_wavelength_overrides.optical.B7=0.740"
            else
                args="+encoder.band_wavelength_overrides.optical.B11=2.190"
                args+=" +encoder.band_wavelength_overrides.optical.B12=1.610"
            fi
            ;;
        smarties_vitb)  # discrete projection index, override dict
            if [ "$pairset" = "s2" ]; then
                args="+encoder.proj_idx_override.optical.B8=11"
                args+=" +encoder.proj_idx_override.optical.B8A=9"
                args+=" +encoder.proj_idx_override.optical.B4=7"
                args+=" +encoder.proj_idx_override.optical.B5=6"
                args+=" +encoder.proj_idx_override.optical.B6=10"
                args+=" +encoder.proj_idx_override.optical.B7=8"
            else
                args="+encoder.proj_idx_override.optical.B11=14"
                args+=" +encoder.proj_idx_override.optical.B12=13"
            fi
            ;;
        *)
            echo "ERROR: unknown encoder '$enc'" >&2
            exit 1
            ;;
    esac

    echo "$args"
}

# -----------------------------------------------------------------------------
# JOB LAUNCHER
# -----------------------------------------------------------------------------
launch_job() {
    local -r enc=$1 ds=$2 task=$3 decoder=$4 preprocessing=$5
    local -r criterion=$6 root=$7 pairset=$8 time_limit=$9

    local -r job_name="p1wl_${enc}_${ds}"
    local -r work_dir="${WORK_BASE}/${enc}"
    local -r log_prefix="${LOG_DIR}/${enc}_${ds}"
    local -r sbatch_script="${job_name}.sh"

    mkdir -p "$work_dir"

    if job_in_queue "$job_name"; then
        printf 'SKIP (queued)  [%-20s | %s]\n' "$enc" "$ds"
        return 0
    fi

    local -a swap
    read -ra swap <<< "$(swap_args "$enc" "$pairset")"

    trap 'rm -f "${sbatch_script}"' EXIT

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${ds}"
    cmd+=" encoder=${enc}"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    cmd+=" finetune=false"
    [ -n "$root" ] && cmd+=" dataset.root_path='${root}'"
    cmd+=" work_dir=${work_dir}"
    cmd+=" use_wandb=True"
    cmd+=" wandb_project=${WANDB_PROJECT}"
    [ "$task" = "linear_classification" ] && cmd+=" batch_size=32"
    cmd+=" ${swap[*]}"

    printf '\nLaunching [%-20s | %s]:\n%s\n' "$enc" "$ds" "$cmd"

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
#SBATCH --output=${log_prefix}_output.%j
#SBATCH --error=${log_prefix}_error.%j
#SBATCH --chdir=/dss/dsshome1/08/di54zop/pangaea-bench_dev

export HYDRA_FULL_ERROR=1
export GEO_BENCH_DIR=${GEO_BENCH_DATA}

module add micromamba
eval "\$(micromamba shell hook --shell bash)"
micromamba activate pangaea-bench

${cmd}
EOF

    chmod +x "$sbatch_script"
    sbatch "$sbatch_script"
    rm "$sbatch_script"
    trap - EXIT
}

# -----------------------------------------------------------------------------
# DATASET DEFINITIONS
# Format: "task decoder preprocessing criterion root_path pairset time_limit"
# preprocessing here is the *default* (external NormalizeMeanStd) variant;
# preprocessing_for_encoder() below swaps in the "_no_norm"/imagewise
# equivalent for carl_imagewise, since it does its own internal per-image
# normalization and must not also get an external NormalizeMeanStd.
# root_path "-" means the dataset config resolves its own path (GeoBench, via
# GEO_BENCH_DIR).
# -----------------------------------------------------------------------------
declare -A datasets=(
    ["mados"]="segmentation seg_upernet seg_focus_crop cross_entropy ${DATASET_PATH}/mados s2 8:00:00"
    ["mcashew-plantation"]="segmentation seg_upernet seg_default cross_entropy - s2 6:00:00"
    ["msa-crop-type"]="segmentation seg_upernet seg_default cross_entropy - s2 6:00:00"
    ["mforestnet"]="linear_classification cls_linear cls_geobench_no_pad cross_entropy - l8 5:00:00"
)

# -----------------------------------------------------------------------------
# preprocessing_for_encoder <enc_key> <base_preprocessing>
# carl_imagewise (image_norm=true, its own per-image normalization) must use
# the "_no_norm" / imagewise preprocessing variant instead of the default
# external-NormalizeMeanStd one. All other encoders keep the default
# preprocessing unchanged.
# -----------------------------------------------------------------------------
preprocessing_for_encoder() {
    local -r enc=$1 base=$2
    if [ "$enc" = "carl_imagewise" ]; then
        case "$base" in
            seg_focus_crop)      echo "seg_focus_crop_no_norm" ;;
            seg_default)         echo "seg_default_no_norm" ;;
            cls_geobench_no_pad) echo "cls_geobench_imagewise_no_pad" ;;
            *)                   echo "$base" ;;
        esac
    else
        echo "$base"
    fi
}

readonly ENCODERS="dofa dofav2 panopticon copernicus_fm_base carl_imagewise smarties_vitb"

RUN_ENCODERS="${RUN_ENCODERS:-all}"
RUN_DATASETS="${RUN_DATASETS:-all}"

for enc in $ENCODERS; do
    if [ "$RUN_ENCODERS" != "all" ]; then
        [[ " $RUN_ENCODERS " != *" $enc "* ]] && continue
    fi

    for ds in "${!datasets[@]}"; do
        if [ "$RUN_DATASETS" != "all" ]; then
            [[ " $RUN_DATASETS " != *" $ds "* ]] && continue
        fi

        read -r task decoder preprocessing criterion root pairset time_limit <<< "${datasets[$ds]}"
        [ "$root" = "-" ] && root=""
        preprocessing=$(preprocessing_for_encoder "$enc" "$preprocessing")

        launch_job "$enc" "$ds" "$task" "$decoder" "$preprocessing" "$criterion" "$root" "$pairset" "$time_limit"
    done
done

echo ""
echo "All P1 wavelength-swap ablation jobs queued (up to 24 total)."
echo "Monitor: squeue -u \$USER --clusters=hpda2"
