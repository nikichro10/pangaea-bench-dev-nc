import wandb
import pandas as pd


"""# Initialize the API
api = wandb.Api()

# Get all runs from a specific project
project_name = "aaron-b-university-of-the-bundeswehr-munich/geofm-bench"
runs = api.runs(project_name)

# Convert to DataFrame - flatten the summary dictionaries
data = []
for run in runs:
    run_dict = {
        'run_id': run.id,
        'name': run.name,
    }
    # Add summary metrics
    run_dict.update(run.summary)
    # Extract the work_dir, the last subdir is the encoder name
    run_dict.update({"work_dir": run.config["work_dir"]})
    run_dict.update({"dataset_name": run.config["dataset"]["dataset_name"]})
    run_dict.update({"limited_label_train": run.config["limited_label_train"]})
    run_dict.update({"learning_rate": run.config["optimizer"]["lr"]})

    # Extract the preprocessing config (e.g. seg_default, seg_default_nonorm, etc.)
    args = run.metadata["args"]
    preprocessing_value = next((arg.split("=", 1)[1] for arg in args if arg.startswith("preprocessing=")), None)
    run_dict.update({"preprocessing": preprocessing_value})
    data.append(run_dict)

df = pd.DataFrame(data)

df.to_csv("scripts/analyze_experiments/df_all_wandb_runs.csv")"""
df = pd.read_csv("scripts/analyze_experiments/df_all_wandb_runs.csv")

# ----------------------------------
# Preprocessing groups
# ----------------------------------

default_preprocessing_values = [
    "seg_default", "seg_resize", "seg_focus_crop"
]

globalnorm_preprocessing_values = [
    "seg_default_customglobalnorm", "seg_resize_customglobalnorm", "seg_focus_crop_customglobalnorm"
]

bandwisenorm_preprocessing_values = [
    "seg_default_custombandwisenorm", "seg_resize_custombandwisenorm", "seg_focus_crop_custombandwisenorm"
]

preprocessing_order = ["Default", "GlobalNorm", "BandwiseNorm"]

datasets = [
    "HLSBurnScars", "Sen1Floods11", "Pastis",
    "MADOS", "AI4SmallFarms", "DynamicEarthNet"
]

models_to_keep = [
    "s2_dino_hiera_bandwise-norm",
    "s2_dino_hiera_global-norm",
    "s2_dino_hiera_specenc_meanagg_bandwise-norm",
    "s2_dino_hiera_specenc_meanagg_global-norm"
]

rename_dataset = {
    "AI4SmallFarms": "AI4Farms",
    "HLSBurnScars": "HLS Burns",
    "MADOS": "MADOS",
    "Pastis": "PASTIS",
    "SN7CD": "SN7",
    "DynamicEarthNet": "DynEarthNet",
    "Sen1Floods11": "Sen1Floods11"
}

# ----------------------------------
# Filtering
# ----------------------------------

df = df[
    df['dataset_name'].isin(datasets) &
    (df['learning_rate'] == 0.0001)
].copy()

df['encoder_name'] = df['work_dir'].apply(
    lambda x: x.split('/')[-1] if isinstance(x, str) else None
)

df = df[df['encoder_name'].isin(models_to_keep)]

# ----------------------------------
# Map preprocessing → 3 groups
# ----------------------------------

def map_preprocessing_group(p):
    if p in default_preprocessing_values:
        return "Default"
    elif p in globalnorm_preprocessing_values:
        return "GlobalNorm"
    elif p in bandwisenorm_preprocessing_values:
        return "BandwiseNorm"
    else:
        return None

df['Preprocessing'] = df['preprocessing'].apply(map_preprocessing_group)
df = df[df['Preprocessing'].notna()]

# ----------------------------------
# Create LaTeX table (forced rows)
# ----------------------------------

def create_latex_table(label_fraction):

    df_label = df[df['limited_label_train'] == label_fraction].copy()

    pivot = df_label.pivot_table(
        index=['encoder_name', 'Preprocessing'],
        columns='dataset_name',
        values='test_mIoU',
        aggfunc='mean'
    )

    # ---- FORCE all 3 preprocessing rows per model ----
    full_index = pd.MultiIndex.from_product(
        [models_to_keep, preprocessing_order],
        names=['encoder_name', 'Preprocessing']
    )

    pivot = pivot.reindex(full_index)

    pivot = pivot.rename(columns=rename_dataset).round(2)

    latex_table = pivot.to_latex(
        float_format="%.2f",
        bold_rows=False,
        multirow=True,
        na_rep="--",
        caption=f"Test mIoU for limited label fraction = {label_fraction}",
        label=f"tab:limited_{str(label_fraction).replace('.', '')}"
    )

    return latex_table


latex_01 = create_latex_table(0.1)
latex_1 = create_latex_table(1.0)

print("===== LATEX TABLE (0.1) =====")
print(latex_01)

print("\n===== LATEX TABLE (1.0) =====")
print(latex_1)