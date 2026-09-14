#!/bin/bash
#
# P2 -- Within-group random wavelength-label permutation ablation.
#
# Follow-up to P1 (single nearly-degenerate pairwise swaps). Where P1 tested
# the sanity floor, P2 tests whether encoders are robust to (and physically
# consistent under) a full random reshuffle of wavelength labels *within* a
# spectrally coherent group -- never letting a label cross into a spectrally
# unrelated group (e.g. a VIS label never gets assigned to a SWIR position).
# As with P1, pixel data and normalization stay completely untouched at their
# original channel position; only the wavelength/proj_idx label fed to each
# encoder's weight generator is permuted.
#
# Spectral groups:
#   S2 11/12-band datasets (MADOS, m-Cashew-Plantation, m-SA-Crop-Type):
#     VIS            {B1, B2, B3, B4}         -- randomly permuted (derangement)
#     Red-edge/NIR   {B5, B6, B7, B8, B8A}    -- randomly permuted (derangement)
#     SWIR           {B11, B12}                -- swapped (only 2 members, so
#                                                  "random permutation" reduces
#                                                  to the deterministic swap;
#                                                  same for both draws)
#     B9 (and B10, not present in these datasets) held fixed -- never touched.
#   m-Forestnet (Landsat-8, 6 bands, S2-aliased names in this dataset config):
#     {blue, green, red} = {B2, B3, B4}        -- randomly permuted (derangement)
#     {SWIR1, SWIR2}     = {B11, B12}          -- swapped (deterministic, as above)
#     NIR (B8) held fixed -- never touched.
#
# All within-group permutations are drawn as *derangements* (no band keeps
# its own wavelength) so every permuted band actually moves -- this maximizes
# signal given only 2 draws. Generated once with numpy, seeds documented
# below, and hardcoded here for full reproducibility/audit:
#
#   SEED_A = 42, SEED_B = 7 (VIS/RGB group), seed+1000 (Red-edge/NIR group),
#   seed+2000 (m-Forestnet RGB group) -- offsets used only to decorrelate the
#   two groups' draws within the same seed, not for any statistical reason.
#   Algorithm: rejection-sample np.random.default_rng(seed).permutation(n)
#   until no fixed point (a true derangement).
#
#   Draw A: VIS  B1->B4 B2->B3 B3->B2 B4->B1 (a pair of 2-swaps)
#           RE/NIR B5->B8 B6->B7 B7->B6 B8->B8A B8A->B5
#           mFN RGB B2->B4 B3->B2 B4->B3 (3-cycle)
#   Draw B: VIS  B1->B2 B2->B3 B3->B4 B4->B1 (4-cycle)
#           RE/NIR B5->B7 B6->B8A B7->B8 B8->B6 B8A->B5
#           mFN RGB B2->B3 B3->B4 B4->B2 (3-cycle)
#
# If the two draws disagree noticeably in the resulting notebook comparison,
# add a third random draw or the deterministic within-group-reversal variant
# (e.g. VIS: B1<->B4, B2<->B3) as a tie-breaker -- not implemented here.
#
# Per-encoder implementation (same mechanism as P1 -- see run_p1_wlswap_ablation.sh):
#   DOFA / DOFAv2 / Panopticon / Copernicus-FM -- plain CLI override on
#     encoder.wave_list.optical.{band} (keys already exist, no '+').
#   CARL     -- encoder.band_wavelength_overrides.optical.{band} (µm), '+' for new keys.
#   SMARTIES -- encoder.proj_idx_override.optical.{band} (discrete projector index), '+' for new keys.
#
# All encoders frozen (finetune=false). W&B project: wavelength-ablation
# (same project as P1; runs are distinguished by job/run name).
#
# 6 encoders x 4 datasets x 2 draws = 48 runs.
#
# Override RUN_ENCODERS / RUN_DATASETS / RUN_DRAWS at call time for partial runs:
#   RUN_ENCODERS="dofa carl_imagewise" RUN_DATASETS="mados" RUN_DRAWS="A" bash run_p2_wlswap_ablation.sh

set -euo pipefail

readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly GEO_BENCH_DATA="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/datasets/geobench"
readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/p2_wlswap_ablation"
readonly WORK_BASE="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/p2_wlswap_ablation"
readonly WANDB_PROJECT="wavelength-ablation"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}" "${WORK_BASE}"

_ACTIVE_JOB_NAMES=$(squeue --clusters=hpda2 -u "$USER" -h --format="%j" 2>/dev/null | tr '\n' '|')
job_in_queue() { echo "|${_ACTIVE_JOB_NAMES}|" | grep -qF "|${1}|"; }

# -----------------------------------------------------------------------------
# swap_args <enc_key> <groupset: s2|l8> <draw: A|B>
# Echoes the Hydra CLI overrides for one encoder's within-group permutation.
# Values are hardcoded (generated offline, see header) -- each is simply the
# encoder's own native wave_list/proj_idx value for the band the permutation
# maps to, so no new numbers are invented, only assignments are reshuffled.
# -----------------------------------------------------------------------------
swap_args() {
    local -r enc=$1 groupset=$2 draw=$3
    local args=""

    case "$enc" in
        dofa)
            if [ "$draw" = "A" ]; then
                if [ "$groupset" = "s2" ]; then
                    args="encoder.wave_list.optical.B1=0.665 encoder.wave_list.optical.B2=0.56 encoder.wave_list.optical.B3=0.49 encoder.wave_list.optical.B4=0.44 encoder.wave_list.optical.B5=0.832 encoder.wave_list.optical.B6=0.783 encoder.wave_list.optical.B7=0.74 encoder.wave_list.optical.B8=0.864 encoder.wave_list.optical.B8A=0.705 encoder.wave_list.optical.B11=2.2 encoder.wave_list.optical.B12=1.61"
                else
                    args="encoder.wave_list.optical.B2=0.665 encoder.wave_list.optical.B3=0.49 encoder.wave_list.optical.B4=0.56 encoder.wave_list.optical.B11=2.2 encoder.wave_list.optical.B12=1.61"
                fi
            else
                if [ "$groupset" = "s2" ]; then
                    args="encoder.wave_list.optical.B1=0.49 encoder.wave_list.optical.B2=0.56 encoder.wave_list.optical.B3=0.665 encoder.wave_list.optical.B4=0.44 encoder.wave_list.optical.B5=0.783 encoder.wave_list.optical.B6=0.864 encoder.wave_list.optical.B7=0.832 encoder.wave_list.optical.B8=0.74 encoder.wave_list.optical.B8A=0.705 encoder.wave_list.optical.B11=2.2 encoder.wave_list.optical.B12=1.61"
                else
                    args="encoder.wave_list.optical.B2=0.56 encoder.wave_list.optical.B3=0.665 encoder.wave_list.optical.B4=0.49 encoder.wave_list.optical.B11=2.2 encoder.wave_list.optical.B12=1.61"
                fi
            fi
            ;;
        dofav2)
            if [ "$draw" = "A" ]; then
                if [ "$groupset" = "s2" ]; then
                    args="encoder.wave_list.optical.B1=0.665 encoder.wave_list.optical.B2=0.56 encoder.wave_list.optical.B3=0.49 encoder.wave_list.optical.B4=0.44 encoder.wave_list.optical.B5=0.832 encoder.wave_list.optical.B6=0.783 encoder.wave_list.optical.B7=0.74 encoder.wave_list.optical.B8=0.864 encoder.wave_list.optical.B8A=0.705 encoder.wave_list.optical.B11=2.2 encoder.wave_list.optical.B12=1.61"
                else
                    args="encoder.wave_list.optical.B2=0.665 encoder.wave_list.optical.B3=0.49 encoder.wave_list.optical.B4=0.56 encoder.wave_list.optical.B11=2.2 encoder.wave_list.optical.B12=1.61"
                fi
            else
                if [ "$groupset" = "s2" ]; then
                    args="encoder.wave_list.optical.B1=0.49 encoder.wave_list.optical.B2=0.56 encoder.wave_list.optical.B3=0.665 encoder.wave_list.optical.B4=0.44 encoder.wave_list.optical.B5=0.783 encoder.wave_list.optical.B6=0.864 encoder.wave_list.optical.B7=0.832 encoder.wave_list.optical.B8=0.74 encoder.wave_list.optical.B8A=0.705 encoder.wave_list.optical.B11=2.2 encoder.wave_list.optical.B12=1.61"
                else
                    args="encoder.wave_list.optical.B2=0.56 encoder.wave_list.optical.B3=0.665 encoder.wave_list.optical.B4=0.49 encoder.wave_list.optical.B11=2.2 encoder.wave_list.optical.B12=1.61"
                fi
            fi
            ;;
        panopticon)
            if [ "$draw" = "A" ]; then
                if [ "$groupset" = "s2" ]; then
                    args="encoder.wave_list.optical.B1=664.63 encoder.wave_list.optical.B2=559.599 encoder.wave_list.optical.B3=492.997 encoder.wave_list.optical.B4=442.923 encoder.wave_list.optical.B5=827.539 encoder.wave_list.optical.B6=782.419 encoder.wave_list.optical.B7=740.552 encoder.wave_list.optical.B8=864.78 encoder.wave_list.optical.B8A=704.006 encoder.wave_list.optical.B11=2203.618 encoder.wave_list.optical.B12=1613.862"
                else
                    args="encoder.wave_list.optical.B2=664.63 encoder.wave_list.optical.B3=492.997 encoder.wave_list.optical.B4=559.599 encoder.wave_list.optical.B11=2203.618 encoder.wave_list.optical.B12=1613.862"
                fi
            else
                if [ "$groupset" = "s2" ]; then
                    args="encoder.wave_list.optical.B1=492.997 encoder.wave_list.optical.B2=559.599 encoder.wave_list.optical.B3=664.63 encoder.wave_list.optical.B4=442.923 encoder.wave_list.optical.B5=782.419 encoder.wave_list.optical.B6=864.78 encoder.wave_list.optical.B7=827.539 encoder.wave_list.optical.B8=740.552 encoder.wave_list.optical.B8A=704.006 encoder.wave_list.optical.B11=2203.618 encoder.wave_list.optical.B12=1613.862"
                else
                    args="encoder.wave_list.optical.B2=559.599 encoder.wave_list.optical.B3=664.63 encoder.wave_list.optical.B4=492.997 encoder.wave_list.optical.B11=2203.618 encoder.wave_list.optical.B12=1613.862"
                fi
            fi
            ;;
        copernicus_fm_base)
            if [ "$draw" = "A" ]; then
                if [ "$groupset" = "s2" ]; then
                    args="encoder.wave_list.optical.B1=665 encoder.wave_list.optical.B2=560 encoder.wave_list.optical.B3=490 encoder.wave_list.optical.B4=443 encoder.wave_list.optical.B5=842 encoder.wave_list.optical.B6=783 encoder.wave_list.optical.B7=740 encoder.wave_list.optical.B8=865 encoder.wave_list.optical.B8A=705 encoder.wave_list.optical.B11=2190 encoder.wave_list.optical.B12=1610"
                else
                    args="encoder.wave_list.optical.B2=665 encoder.wave_list.optical.B3=490 encoder.wave_list.optical.B4=560 encoder.wave_list.optical.B11=2190 encoder.wave_list.optical.B12=1610"
                fi
            else
                if [ "$groupset" = "s2" ]; then
                    args="encoder.wave_list.optical.B1=490 encoder.wave_list.optical.B2=560 encoder.wave_list.optical.B3=665 encoder.wave_list.optical.B4=443 encoder.wave_list.optical.B5=783 encoder.wave_list.optical.B6=865 encoder.wave_list.optical.B7=842 encoder.wave_list.optical.B8=740 encoder.wave_list.optical.B8A=705 encoder.wave_list.optical.B11=2190 encoder.wave_list.optical.B12=1610"
                else
                    args="encoder.wave_list.optical.B2=560 encoder.wave_list.optical.B3=665 encoder.wave_list.optical.B4=490 encoder.wave_list.optical.B11=2190 encoder.wave_list.optical.B12=1610"
                fi
            fi
            ;;
        carl_imagewise)
            if [ "$draw" = "A" ]; then
                if [ "$groupset" = "s2" ]; then
                    args="+encoder.band_wavelength_overrides.optical.B1=0.665 +encoder.band_wavelength_overrides.optical.B2=0.56 +encoder.band_wavelength_overrides.optical.B3=0.49 +encoder.band_wavelength_overrides.optical.B4=0.443 +encoder.band_wavelength_overrides.optical.B5=0.842 +encoder.band_wavelength_overrides.optical.B6=0.783 +encoder.band_wavelength_overrides.optical.B7=0.74 +encoder.band_wavelength_overrides.optical.B8=0.865 +encoder.band_wavelength_overrides.optical.B8A=0.705 +encoder.band_wavelength_overrides.optical.B11=2.19 +encoder.band_wavelength_overrides.optical.B12=1.61"
                else
                    args="+encoder.band_wavelength_overrides.optical.B2=0.665 +encoder.band_wavelength_overrides.optical.B3=0.49 +encoder.band_wavelength_overrides.optical.B4=0.56 +encoder.band_wavelength_overrides.optical.B11=2.19 +encoder.band_wavelength_overrides.optical.B12=1.61"
                fi
            else
                if [ "$groupset" = "s2" ]; then
                    args="+encoder.band_wavelength_overrides.optical.B1=0.49 +encoder.band_wavelength_overrides.optical.B2=0.56 +encoder.band_wavelength_overrides.optical.B3=0.665 +encoder.band_wavelength_overrides.optical.B4=0.443 +encoder.band_wavelength_overrides.optical.B5=0.783 +encoder.band_wavelength_overrides.optical.B6=0.865 +encoder.band_wavelength_overrides.optical.B7=0.842 +encoder.band_wavelength_overrides.optical.B8=0.74 +encoder.band_wavelength_overrides.optical.B8A=0.705 +encoder.band_wavelength_overrides.optical.B11=2.19 +encoder.band_wavelength_overrides.optical.B12=1.61"
                else
                    args="+encoder.band_wavelength_overrides.optical.B2=0.56 +encoder.band_wavelength_overrides.optical.B3=0.665 +encoder.band_wavelength_overrides.optical.B4=0.49 +encoder.band_wavelength_overrides.optical.B11=2.19 +encoder.band_wavelength_overrides.optical.B12=1.61"
                fi
            fi
            ;;
        smarties_vitb)
            if [ "$draw" = "A" ]; then
                if [ "$groupset" = "s2" ]; then
                    args="+encoder.proj_idx_override.optical.B1=6 +encoder.proj_idx_override.optical.B2=4 +encoder.proj_idx_override.optical.B3=1 +encoder.proj_idx_override.optical.B4=0 +encoder.proj_idx_override.optical.B5=9 +encoder.proj_idx_override.optical.B6=10 +encoder.proj_idx_override.optical.B7=8 +encoder.proj_idx_override.optical.B8=11 +encoder.proj_idx_override.optical.B8A=7 +encoder.proj_idx_override.optical.B11=14 +encoder.proj_idx_override.optical.B12=13"
                else
                    args="+encoder.proj_idx_override.optical.B2=6 +encoder.proj_idx_override.optical.B3=1 +encoder.proj_idx_override.optical.B4=4 +encoder.proj_idx_override.optical.B11=14 +encoder.proj_idx_override.optical.B12=13"
                fi
            else
                if [ "$groupset" = "s2" ]; then
                    args="+encoder.proj_idx_override.optical.B1=1 +encoder.proj_idx_override.optical.B2=4 +encoder.proj_idx_override.optical.B3=6 +encoder.proj_idx_override.optical.B4=0 +encoder.proj_idx_override.optical.B5=10 +encoder.proj_idx_override.optical.B6=11 +encoder.proj_idx_override.optical.B7=9 +encoder.proj_idx_override.optical.B8=8 +encoder.proj_idx_override.optical.B8A=7 +encoder.proj_idx_override.optical.B11=14 +encoder.proj_idx_override.optical.B12=13"
                else
                    args="+encoder.proj_idx_override.optical.B2=4 +encoder.proj_idx_override.optical.B3=6 +encoder.proj_idx_override.optical.B4=1 +encoder.proj_idx_override.optical.B11=14 +encoder.proj_idx_override.optical.B12=13"
                fi
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
    local -r criterion=$6 root=$7 groupset=$8 draw=$9 time_limit=${10}

    local -r job_name="p2wl_${enc}_${ds}_${draw}"
    local -r work_dir="${WORK_BASE}/${enc}_draw${draw}"
    local -r log_prefix="${LOG_DIR}/${enc}_${ds}_${draw}"
    local -r sbatch_script="${job_name}.sh"

    mkdir -p "$work_dir"

    if job_in_queue "$job_name"; then
        printf 'SKIP (queued)  [%-20s | %-20s | draw %s]\n' "$enc" "$ds" "$draw"
        return 0
    fi

    local -a swap
    read -ra swap <<< "$(swap_args "$enc" "$groupset" "$draw")"

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

    printf '\nLaunching [%-20s | %-20s | draw %s]:\n%s\n' "$enc" "$ds" "$draw" "$cmd"

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
# Format: "task decoder preprocessing criterion root_path groupset time_limit"
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
readonly DRAWS="A B"

RUN_ENCODERS="${RUN_ENCODERS:-all}"
RUN_DATASETS="${RUN_DATASETS:-all}"
RUN_DRAWS="${RUN_DRAWS:-all}"

for enc in $ENCODERS; do
    if [ "$RUN_ENCODERS" != "all" ]; then
        [[ " $RUN_ENCODERS " != *" $enc "* ]] && continue
    fi

    for ds in "${!datasets[@]}"; do
        if [ "$RUN_DATASETS" != "all" ]; then
            [[ " $RUN_DATASETS " != *" $ds "* ]] && continue
        fi

        read -r task decoder preprocessing criterion root groupset time_limit <<< "${datasets[$ds]}"
        [ "$root" = "-" ] && root=""
        preprocessing=$(preprocessing_for_encoder "$enc" "$preprocessing")

        for draw in $DRAWS; do
            if [ "$RUN_DRAWS" != "all" ]; then
                [[ " $RUN_DRAWS " != *" $draw "* ]] && continue
            fi
            launch_job "$enc" "$ds" "$task" "$decoder" "$preprocessing" "$criterion" "$root" "$groupset" "$draw" "$time_limit"
        done
    done
done

echo ""
echo "All P2 wavelength-permutation ablation jobs queued (up to 48 total)."
echo "Monitor: squeue -u \$USER --clusters=hpda2"
