#!/bin/bash
#
# P6 -- Fully random wavelength-label ablation, SMARTIES discrete analog.
#
# SMARTIES has no continuous wavelength input at all -- each band name is
# mapped through a fixed lookup table onto one of its spectral-range linear
# projectors (see pangaea/encoders/smarties_encoder.py, _BAND_TO_PROJ_IDX /
# proj_idx_override). A literal "random float from Uniform(400,2500)" (what
# the other 5 encoders' P6 condition draws) has no equivalent input SMARTIES
# can consume. This is the closest discrete analog: each band gets an
# independent *random projector index*, drawn with the exact same two seeds
# (501, 502) and per-dataset-group offsets as the continuous P6 draws in
# p6_random_wl_draws.json, so it "falls into a random bin" instead of a
# random wavelength -- same spirit, same seeds, different (discrete) output
# space forced by the architecture.
#
# Candidate pool: SMARTIES's 15 *optical* projector indices (0-14, aerosol
# through SWIR-2190nm) -- excludes -1 (no projector), the microwave indices
# (15/16, SAR-only), the thermal index (17), and the Landsat-8 blue aggregate
# (18), i.e. the optical-only subset that parallels Uniform(400,2500nm)'s
# VIS->SWIR span for the other encoders.
#
# Sampling: rng = np.random.default_rng(seed + offset), rejection-sampled so
# no two bands within one draw's vector land on the same projector index
# (discrete analog of the >=5nm spacing guard used for the continuous draws).
# Computed once offline; the literal values below are the output of that
# script, not re-derived at submit time.
#
# Output lands in the SAME folders as the other 5 encoders' P6 runs
# (p6_random_wl_ablation/smarties_vitb_draw{1,2}/,
# p6_random_wl_ablation_upernet/smarties_vitb_draw{1,2}/) so
# analyze_wrong_wavelength_summary.ipynb picks it up on the next re-run with
# zero notebook changes -- its P6 loader already iterates smarties_vitb, it
# just never found anything there before.
#
# SpaceNet7-CD uses criterion=weighted_bce_sqrt + best_metric_key=IoU +
# data_replicate=2 from the start (the dice-collapse fix already rolled out
# to the other 5 encoders' P6 SpaceNet7-CD legs this session), not the
# original P6 script's criterion=binary_cross_entropy.
#
# Control via env vars:
#   RUN_JOBS="all"   or space-separated "<dataset>_<family>_<draw>" keys
#                     (e.g. "mados_mlp_1 spacenet7cd_upernet_2")
# -----------------------------------------------------------------------------

readonly DATASET_PATH="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/pangaea-datasets"
readonly GEO_BENCH_DATA="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/datasets/geobench"
readonly LOG_DIR="/dss/dsshome1/08/di54zop/pangaea-bench_dev/scripts/logs/p6_smarties_random_bins"
readonly WORK_BASE_MLP="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/p6_random_wl_ablation"
readonly WORK_BASE_UP="/dss/dsstbyfs02/pn49cu/pn49cu-dss-0001/niklas/experiments/p6_random_wl_ablation_upernet"
readonly SLURM_EXCLUDE="hpdar04c04s02,hpdar04c02s05,hpdar04c01s01,hpdar04c01s02,hpdar04c01s03,hpdar04c03s04,hpdar04c03s05,hpdar04c05s03,hpdar04c05s05,hpdar04c06s02"

mkdir -p "${LOG_DIR}"

_ACTIVE_JOB_NAMES=$(squeue --clusters=hpda2 -u "$USER" -h --format="%j" 2>/dev/null | tr '\n' '|')
job_in_queue() { echo "|${_ACTIVE_JOB_NAMES}|" | grep -qF "|${1}|"; }

# -----------------------------------------------------------------------------
# proj_idx_args <dsgroup: mados|s2_12b|mfn|sn7cd|fbp> <draw: 1|2>
# Echoes the +encoder.proj_idx_override.optical.{band}={idx} CLI overrides.
# Literal values from the offline RNG draw described above.
# -----------------------------------------------------------------------------
proj_idx_args() {
    local -r dsgroup=$1 draw=$2
    case "$dsgroup" in
        mados)
            case "$draw" in
                1) echo "+encoder.proj_idx_override.optical.B1=8 +encoder.proj_idx_override.optical.B2=3 +encoder.proj_idx_override.optical.B3=13 +encoder.proj_idx_override.optical.B4=14 +encoder.proj_idx_override.optical.B5=2 +encoder.proj_idx_override.optical.B6=1 +encoder.proj_idx_override.optical.B7=11 +encoder.proj_idx_override.optical.B8=9 +encoder.proj_idx_override.optical.B8A=6 +encoder.proj_idx_override.optical.B11=7 +encoder.proj_idx_override.optical.B12=12" ;;
                2) echo "+encoder.proj_idx_override.optical.B1=10 +encoder.proj_idx_override.optical.B2=2 +encoder.proj_idx_override.optical.B3=11 +encoder.proj_idx_override.optical.B4=3 +encoder.proj_idx_override.optical.B5=9 +encoder.proj_idx_override.optical.B6=8 +encoder.proj_idx_override.optical.B7=12 +encoder.proj_idx_override.optical.B8=0 +encoder.proj_idx_override.optical.B8A=5 +encoder.proj_idx_override.optical.B11=14 +encoder.proj_idx_override.optical.B12=7" ;;
            esac ;;
        s2_12b)
            case "$draw" in
                1) echo "+encoder.proj_idx_override.optical.B1=14 +encoder.proj_idx_override.optical.B2=2 +encoder.proj_idx_override.optical.B3=7 +encoder.proj_idx_override.optical.B4=6 +encoder.proj_idx_override.optical.B5=11 +encoder.proj_idx_override.optical.B6=0 +encoder.proj_idx_override.optical.B7=12 +encoder.proj_idx_override.optical.B8=4 +encoder.proj_idx_override.optical.B8A=13 +encoder.proj_idx_override.optical.B9=3 +encoder.proj_idx_override.optical.B11=10 +encoder.proj_idx_override.optical.B12=1" ;;
                2) echo "+encoder.proj_idx_override.optical.B1=11 +encoder.proj_idx_override.optical.B2=7 +encoder.proj_idx_override.optical.B3=0 +encoder.proj_idx_override.optical.B4=14 +encoder.proj_idx_override.optical.B5=5 +encoder.proj_idx_override.optical.B6=6 +encoder.proj_idx_override.optical.B7=13 +encoder.proj_idx_override.optical.B8=3 +encoder.proj_idx_override.optical.B8A=10 +encoder.proj_idx_override.optical.B9=1 +encoder.proj_idx_override.optical.B11=4 +encoder.proj_idx_override.optical.B12=8" ;;
            esac ;;
        mfn)
            case "$draw" in
                1) echo "+encoder.proj_idx_override.optical.B2=8 +encoder.proj_idx_override.optical.B3=6 +encoder.proj_idx_override.optical.B4=14 +encoder.proj_idx_override.optical.B8=10 +encoder.proj_idx_override.optical.B11=1 +encoder.proj_idx_override.optical.B12=7" ;;
                2) echo "+encoder.proj_idx_override.optical.B2=2 +encoder.proj_idx_override.optical.B3=14 +encoder.proj_idx_override.optical.B4=0 +encoder.proj_idx_override.optical.B8=8 +encoder.proj_idx_override.optical.B11=12 +encoder.proj_idx_override.optical.B12=9" ;;
            esac ;;
        sn7cd)
            case "$draw" in
                1) echo "+encoder.proj_idx_override.optical.B4=3 +encoder.proj_idx_override.optical.B3=6 +encoder.proj_idx_override.optical.B2=13" ;;
                2) echo "+encoder.proj_idx_override.optical.B4=3 +encoder.proj_idx_override.optical.B3=1 +encoder.proj_idx_override.optical.B2=13" ;;
            esac ;;
        fbp)
            case "$draw" in
                1) echo "+encoder.proj_idx_override.optical.B8=0 +encoder.proj_idx_override.optical.B4=9 +encoder.proj_idx_override.optical.B3=12 +encoder.proj_idx_override.optical.B2=8" ;;
                2) echo "+encoder.proj_idx_override.optical.B8=2 +encoder.proj_idx_override.optical.B4=11 +encoder.proj_idx_override.optical.B3=9 +encoder.proj_idx_override.optical.B2=4" ;;
            esac ;;
    esac
}

# -----------------------------------------------------------------------------
# JOB LAUNCHER
# -----------------------------------------------------------------------------
launch_job() {
    local -r ds=$1 task=$2 decoder=$3 preprocessing=$4 criterion=$5
    local -r root=$6 dsgroup=$7 draw=$8 time_limit=$9 decoder_family=${10}
    shift 10
    local -ra extra_args=("$@")

    local -r job_name="p6smbin_${ds:0:10}_${decoder_family}_${draw}"
    local -r work_dir="$([ "$decoder_family" = "mlp" ] && echo "${WORK_BASE_MLP}/smarties_vitb_draw${draw}" || echo "${WORK_BASE_UP}/smarties_vitb_draw${draw}")"
    local -r log_prefix="${LOG_DIR}/${ds}_${decoder_family}_${draw}"
    local -r sbatch_script="${job_name}.sh"

    mkdir -p "$work_dir"

    if job_in_queue "$job_name"; then
        printf 'SKIP (queued)  [%-20s | %-8s | draw %s]\n' "$ds" "$decoder_family" "$draw"
        return 0
    fi

    local -a proj_idx
    read -ra proj_idx <<< "$(proj_idx_args "$dsgroup" "$draw")"

    trap 'rm -f "${sbatch_script}"' EXIT

    local cmd="torchrun --standalone --nnodes=1 --nproc_per_node=1 pangaea/run.py"
    cmd+=" --config-name=train"
    cmd+=" dataset=${ds}"
    cmd+=" encoder=smarties_vitb"
    cmd+=" decoder=${decoder}"
    cmd+=" preprocessing=${preprocessing}"
    cmd+=" criterion=${criterion}"
    cmd+=" task=${task}"
    [ -n "$root" ] && cmd+=" dataset.root_path='${root}'"
    cmd+=" work_dir=${work_dir}"
    cmd+=" use_wandb=True"
    cmd+=" wandb_project=wavelength-ablation"
    cmd+=" finetune=false"
    [ "$task" = "linear_classification" ] && cmd+=" batch_size=32"
    if [ "$ds" = "spacenet7cd" ]; then
        cmd+=" task.trainer.best_metric_key=IoU data_replicate=2"
    fi
    cmd+=" ${proj_idx[*]}"
    if [ "${#extra_args[@]}" -gt 0 ]; then
        cmd+=" ${extra_args[*]}"
    fi

    printf '\nLaunching [%-20s | %-8s | draw %s]:\n%s\n' "$ds" "$decoder_family" "$draw" "$cmd"

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
export WANDB_MODE=offline
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
# Format per decoder family: "task decoder preprocessing criterion root_path dsgroup time_limit"
# spacenet7cd's criterion is fixed below (weighted_bce_sqrt), not read from here.
# -----------------------------------------------------------------------------
declare -A datasets_mlp=(
    ["mados"]="segmentation seg_mlp_fusion seg_focus_crop cross_entropy ${DATASET_PATH}/mados mados 8:00:00"
    ["mcashew-plantation"]="segmentation seg_mlp_fusion seg_default cross_entropy - s2_12b 6:00:00"
    ["msa-crop-type"]="segmentation seg_mlp_fusion seg_default cross_entropy - s2_12b 6:00:00"
    ["mforestnet"]="linear_classification cls_linear cls_geobench_no_pad cross_entropy - mfn 5:00:00"
    ["spacenet7cd"]="change_detection seg_siam_mlp_fusion seg_default weighted_bce_sqrt ${DATASET_PATH}/spacenet7 sn7cd 8:00:00"
    ["fivebillionpixels"]="segmentation seg_mlp_fusion seg_default cross_entropy ${DATASET_PATH}/fivebillionpixels fbp 16:00:00"
)

declare -A datasets_upernet=(
    ["mados"]="segmentation seg_upernet seg_focus_crop cross_entropy ${DATASET_PATH}/mados mados 8:00:00"
    ["mcashew-plantation"]="segmentation seg_upernet seg_default cross_entropy - s2_12b 6:00:00"
    ["msa-crop-type"]="segmentation seg_upernet seg_default cross_entropy - s2_12b 6:00:00"
    ["mforestnet"]="linear_classification cls_linear cls_geobench_no_pad cross_entropy - mfn 5:00:00"
    ["spacenet7cd"]="change_detection seg_siamupernet_conc seg_default weighted_bce_sqrt ${DATASET_PATH}/spacenet7 sn7cd 8:00:00"
    ["fivebillionpixels"]="segmentation seg_upernet seg_default cross_entropy ${DATASET_PATH}/fivebillionpixels fbp 16:00:00"
)

RUN_JOBS="${RUN_JOBS:-all}"

for decoder_family in mlp upernet; do
    if [ "$decoder_family" = "mlp" ]; then
        declare -n datasets_ref=datasets_mlp
    else
        declare -n datasets_ref=datasets_upernet
    fi

    for ds in "${!datasets_ref[@]}"; do
        read -r task decoder preprocessing criterion root dsgroup time_limit <<< "${datasets_ref[$ds]}"
        [ "$root" = "-" ] && root=""

        for draw in 1 2; do
            combo_key="${ds}_${decoder_family}_${draw}"
            if [ "$RUN_JOBS" != "all" ]; then
                [[ " $RUN_JOBS " != *" $combo_key "* ]] && continue
            fi
            launch_job "$ds" "$task" "$decoder" "$preprocessing" "$criterion" "$root" "$dsgroup" "$draw" "$time_limit" "$decoder_family"
        done
    done
done

echo ""
echo "All SMARTIES P6 random-bin jobs queued (up to 24 total)."
echo "Monitor: squeue -u \$USER --clusters=hpda2"
