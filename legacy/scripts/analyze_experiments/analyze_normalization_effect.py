import wandb
import pandas as pd


# Initialize the API
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

#df.to_csv("scripts/analyze_experiments/df_all_wandb_runs.csv")
#df = pd.read_csv("scripts/analyze_experiments/df_all_wandb_runs.csv")

# List of values to keep
valid_preprocessing_values = ["seg_default_nonorm", "seg_resize_nonorm", "seg_focus_crop_nonorm"]

# Filter the DataFrame
df_filtered = df[df['preprocessing'].isin(valid_preprocessing_values)]

# Only keep the runs on reproducible datasets
datasets = ["HLSBurnScars", "Sen1Floods11", "Pastis", "MADOS", "SN7CD", "AI4SmallFarms", "DynamicEarthNet"]
df_filtered = df_filtered[df_filtered['dataset_name'].isin(datasets)]
df_filtered = df_filtered[df_filtered['learning_rate'] == 0.0001]

for limited_label_fraction in [0.1, 1.0]:
    # Create a df that only contains the label fraction of each scenario
    df_label_fraction = df_filtered[df_filtered['limited_label_train'] == limited_label_fraction].copy()

    # Create one column with the encoder name, it's the last subdir of the work_dir path
    df_label_fraction['encoder_name'] = df_label_fraction['work_dir'].apply(
        lambda x: x.split('/')[-1] if pd.notna(x) and isinstance(x, str) else None)

    # Create the pivot table once wandb is cleared up (no duplicate runs)
    pivot_table = df_label_fraction.pivot(index='encoder_name', columns='dataset_name', values='test_mIoU').round(2)

    # Define the desired order and reorder rows
    desired_order = ["remoteclip", "prithvi", "s2_mae", "s2_dino", "s2_dinov2", "s2_ijepa", "l8_mae", "l8_dinov2", "l8_ijepa"]
    pivot_table = pivot_table.reindex(desired_order)

    # Define the renaming mapping and rename the columns (Datasets)
    rename_mapping = {
        "AI4SmallFarms": "AI4Farms",
        "HLSBurnScars": "HLS Burns", 
        "MADOS": "MADOS",
        "Pastis": "PASTIS", 
        "SN7CD": "SN7",
        "DynamicEarthNet": "DynEarthNet",
        "Sen1Floods11": "Sen1Floods11"  # Note: corrected the key name
    }
    pivot_table = pivot_table.rename(columns=rename_mapping)

    # Define the renaming mapping and rename the columns (Datasets)
    rename_mapping = {
        "remoteclip": "RemoteCLIP repro",
        "S12-Data2Vec": "S12-Data2Vec repro",
        "prithvi": "Prithvi repro",
        "s2_mae": "S2-MAE",
        "s2_dino": "S2-DINO",
        "s2_dinov2": "S2-DINOv2",
        "s2_ijepa": "S2-iJEPA",
        "l8_mae": "L8-MAE",
        "l8_dinov2": "L8-DINOv2",
        "l8_ijepa": "L8-iJEPA",
    }
    pivot_table = pivot_table.rename(index=rename_mapping)

    # Rename encoder_name to Model (currently "index", not)
    pivot_table = pivot_table.reset_index().rename(columns={'encoder_name': 'Model'})

    # Read pangaea results csv (including TerraMind and Thor)
    df_pangaea = pd.read_csv(f'scripts/analyze_experiments/pangaea_results_{limited_label_fraction}.csv', sep=' ')
    # Drop multiple columns
    df_pangaea = df_pangaea.drop(['FBP', 'CropMap'], axis=1)

    # Now merge
    merged_df = pd.concat([df_pangaea, pivot_table], ignore_index=True)
    print(merged_df)

    latex_table = merged_df.to_latex(
        caption=f"Extended Pangaea results with {int(limited_label_fraction*100)}\% training data in mIoU",
        label="tab:pangaea_benchmark",
        float_format='%.2f',
        index=False,  # Include row indices
    )
    print(latex_table)