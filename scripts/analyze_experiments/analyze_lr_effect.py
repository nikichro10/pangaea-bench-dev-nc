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

# There are experiments with ablations on other preprocessing (no normalization)
# so only keep the pangaea default preprocessing for this analysis
valid_preprocessing_values = ["seg_default", "seg_resize", "seg_focus_crop"]
df_filtered = df[df['preprocessing'].isin(valid_preprocessing_values)]

# Only keep the runs on reproducible datasets, LR analysis only done on 3 datasets
datasets = ["HLSBurnScars", "Sen1Floods11", "MADOS"]
df_filtered = df_filtered[df_filtered['dataset_name'].isin(datasets)]

# Create one column with the encoder name, it's the last subdir of the work_dir path
df_filtered['encoder_name'] = df_filtered['work_dir'].apply(
    lambda x: x.split('/')[-1] if pd.notna(x) and isinstance(x, str) else None)

encoders_lr_analysis = ["terramind_base", "s2_dino_hiera", "l8_dino_hiera", "s2_dino_hiera_per-band", "s2_dino_hiera_semantic"]
# Filter for encoders we're interested in
df_filtered = df_filtered[df_filtered['encoder_name'].isin(encoders_lr_analysis)]

for limited_label_fraction in [1.0]:
    # Create a df that only contains the label fraction of each scenario
    df_label_fraction = df_filtered[df_filtered['limited_label_train'] == limited_label_fraction].copy()
    #print(df_label_fraction) # 6 lrs x 3 datasets x 5 encoders = 90 rows

    # Initialize result dictionary
    results = {
        'encoder_name': encoders_lr_analysis
    }
    
    # For each dataset, get default and optimal values
    for dataset in datasets:
        df_dataset = df_label_fraction[df_label_fraction['dataset_name'] == dataset]
        print(df_dataset)
        
        # Default value: test_mIoU where learning_rate == 0.0001
        df_default = df_dataset[df_dataset['learning_rate'] == 0.0001]

        default_values = {}
        
        for encoder in encoders_lr_analysis:
            encoder_default = df_default[df_default['encoder_name'] == encoder]
            if not encoder_default.empty:
                default_values[encoder] = encoder_default['test_mIoU'].iloc[0]
            else:
                default_values[encoder] = None
        
        # Optimal value: best test_mIoU across all learning rates
        optimal_values = {}
        for encoder in encoders_lr_analysis:
            encoder_all_lr = df_dataset[df_dataset['encoder_name'] == encoder]
            
            if not encoder_all_lr.empty:
                optimal_values[encoder] = encoder_all_lr['test_mIoU'].max()
            else:
                optimal_values[encoder] = None
        
        # Add to results dictionary
        results[f'{dataset} default'] = [default_values.get(encoder) for encoder in encoders_lr_analysis]
        results[f'{dataset} optimal'] = [optimal_values.get(encoder) for encoder in encoders_lr_analysis]
    
    # Create DataFrame and display
    results_df = pd.DataFrame(results)
    
    # Format to 2 decimal places for better readability
    for col in results_df.columns[1:]:  # Skip encoder_name column
        results_df[col] = results_df[col].apply(lambda x: round(x, 2) if x is not None else None)

    # Define the renaming mapping and rename the rows (Encoders)
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
        "s2_mae_hiera": "S2-MAE Hiera",
        "l8_mae_hiera": "L8-MAE Hiera",
        "s2_dino_hiera": "S2-DINO Hiera",
        "l8_dino_hiera": "L8-DINO Hiera",
        "terramind_base": "TerraMind-B",
        "s2_dino_hiera_per-band": "S2-DINO Hiera per-band",
        "s2_dino_hiera_semantic": "S2-DINO Hiera semantic",
        "s2_ijepa_hiera_semantic": "S2-iJEPA Hiera semantic",
        "l8_dino_hiera_semantic": "L8-DINO Hiera semantic",
        "l8_dino_hiera_per-band": "L8-DINO Hiera per-band",
        "s2_mae_hiera_semantic": "S2-MAE Hiera semantic",
        "l8_mae_hiera_semantic": "L8-MAE Hiera semantic",
    }
    print(results_df)
    results_df['encoder_name'] = results_df['encoder_name'].replace(rename_mapping)
    
    latex_table = results_df.to_latex(
        caption=f"Pangaea LR results with {int(limited_label_fraction*100)}\% training data in mIoU",
        label="tab:pangaea_lr",
        float_format='%.2f',
        index=False,  # Include row indices
    )
    print(latex_table)
