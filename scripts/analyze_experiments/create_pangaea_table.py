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
    # Extract the work_dir, the last subdir of the work_dir contains the encoder name
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

df.to_csv("scripts/analyze_experiments/df_all_wandb_runs.csv")
#exit()
df = pd.read_csv("scripts/analyze_experiments/df_all_wandb_runs.csv")

# There are experiments with ablations on other preprocessing (no normalization)
# And different learning rates, so only keep the pangaea default values for this table
valid_preprocessing_values = ["seg_default", "seg_resize", "seg_focus_crop"]
df_filtered = df[df['preprocessing'].isin(valid_preprocessing_values)]
df_filtered = df_filtered[df_filtered['learning_rate'] == 0.0001]

# Only keep the runs on reproducible datasets
datasets = ["HLSBurnScars", "Sen1Floods11", "Pastis", "MADOS", "SN7CD", "AI4SmallFarms", "DynamicEarthNet"]
df_filtered = df_filtered[df_filtered['dataset_name'].isin(datasets)]

for limited_label_fraction in [0.1, 1.0]:
    # Create a df that only contains the label fraction of each scenario
    df_label_fraction = df_filtered[df_filtered['limited_label_train'] == limited_label_fraction].copy()

    # Create one column with the encoder name, it's the last subdir of the work_dir path
    df_label_fraction['encoder_name'] = df_label_fraction['work_dir'].apply(lambda x: x.split('/')[-1] if pd.notna(x) and isinstance(x, str) else None)

    # Create the pivot table once wandb is cleared up (no duplicate runs)
    pivot_table = df_label_fraction.pivot(index='encoder_name', columns='dataset_name', values='test_mIoU')
    pivot_table = pivot_table.round(2)

    # Define the desired order and reorder rows
    desired_order = ["vit12_rgb_random_init",
                     "vit8_12bands_random_init",
                     "vit16_12bands_random_init_224input",
                     "vit12_12bands_random_init",
                     "vit16_12bands_random_init",
                     "vit16_12bands_random_init_finetuned",
                     #"remoteclip", "prithvi", "terramind_base",
                     "s2_mae", "s2_dino", "s2_dinov2", "s2_ijepa",
                     "l8_mae", "l8_dinov2", "l8_ijepa",
                     "s2_mae_hiera", "s2_dino_hiera",
                     "l8_dino_hiera", "l8_mae_hiera",
                     #"s2_dino_hiera_per-band", "s2_dino_hiera_semantic", "s2_ijepa_hiera_semantic",
                     #"l8_dino_hiera_semantic", "l8_dino_hiera_per-band",
                     #"s2_mae_hiera_semantic", "l8_mae_hiera_semantic",
                     "s2_dino_hiera_bandwise-norm", "s2_dino_hiera_global-norm",
                     "spe_mm_linearPE", "spe_mm_dino_v2_L8", "spe_mm_dinov2", "spe_mm_dinov2_v2",
                     "s2_dino_hiera_feb26", "s2_dino_hiera_feb26_v2", "spe_mm_dino_v1", "spe_mm_dino_v2",
                     "s2_dino_hieraB_4stage_patchs3", "s2_dino_hieraL_3stage_patchs6", "s2_dino_hieraL_4stage_patchs3",
                     "hierab_mm_dino_3stages",
                     "hierab_mm_dino_3stages_S2S1",
                     "hierab_mm_dino_3stages_rescales2",
                     "hierab_mm_dino_3stages_rescales4",
                     "hierab_mm_dino_3stages_rescales421",
                     "hierab_mm_dinov2_3stages",
                     "hierab_mm_dinov2_3stages_S2S1",
                     "hierab_s2_jepa_3stages",
                     "hierab_s2_jepa_3stages_lossweighting",
                     "s2_dino_hiera_lowbanddrop",
                     "s2_dino_hiera_mediumbanddrop",
                     "s2_dino_hiera_highbanddrop",
                     "hierab_s2_jepa_lowbanddrop",
                     "hierab_s2_jepa_mediumbanddrop",
                     "hierab_s2_jepa_lowlambdadetach",
                     "hierab_s2_jepa_lowlambda",
                     "hierab_s2_jepa_verylowlambda",
                     "hierab_s2_jepa_highlambda",
                     "hierab_s2_jepa_highlambdadetach",
                     "hierab_mm_jepa_highlambda",
                     "hierab_mm_jepa_highlambdadetach",
                     "hierab_mm_jepa_lowlambdadetach",
                     #"spe_mm_jepa_wospecenc",
                     ]
    pivot_table = pivot_table.reindex(desired_order)

    # Define the renaming mapping and rename the columns (Datasets)
    rename_mapping = {
        "AI4SmallFarms": "AI4Farms",
        "HLSBurnScars": "HLS Burns", 
        "MADOS": "MADOS",
        "Pastis": "PASTIS", 
        "SN7CD": "SN7",
        "DynamicEarthNet": "DynEarthNet",
        "Sen1Floods11": "Sen1Floods11"
    }
    pivot_table = pivot_table.rename(columns=rename_mapping)

    # Define the renaming mapping and rename the rows (Encoders)
    rename_mapping = {
        "vit12_rgb_random_init": "ViT12 RGB random-init",
        "vit16_12bands_random_init_224input": "ViT16 12Bands random-init 224input",
        "vit8_12bands_random_init": "ViT8 12Bands random-init",
        "vit12_12bands_random_init": "ViT12 12Bands random-init",
        "vit16_12bands_random_init": "ViT16 12Bands random-init",
        "vit16_12bands_random_init_finetuned": "ViT16 12Bands random-init finetuned",
        #"remoteclip": "RemoteCLIP repro",
        #"prithvi": "Prithvi repro",
        #"terramind_base": "TerraMind-B repro",
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
        #"s2_dino_hiera_per-band": "S2-DINO Hiera per-band",
        #"s2_dino_hiera_semantic": "S2-DINO Hiera semantic",
        #"s2_ijepa_hiera_semantic": "S2-iJEPA Hiera semantic",
        #"l8_dino_hiera_semantic": "L8-DINO Hiera semantic",
        #"l8_dino_hiera_per-band": "L8-DINO Hiera per-band",
        #"s2_mae_hiera_semantic": "S2-MAE Hiera semantic",
        #"l8_mae_hiera_semantic": "L8-MAE Hiera semantic",
        "s2_dino_hiera_bandwise-norm": "S2-DINO Hiera bw-norm",
        "s2_dino_hiera_global-norm": "S2-DINO Hiera gl-norm",
        "spe_mm_linearPE": "spe-mm-linearPE",
        "spe_mm_dino_v2_L8": "spe-mm-dinov2 L8",
        "spe_mm_dinov2": "spe-mm-dinov2",
        "spe_mm_dinov2_v2": "spe-mm-dinov2-v2",
        "s2_dino_hiera_feb26": "s2-dino-hiera-feb26",
        "s2_dino_hiera_feb26_v2": "s2-dino-hiera-feb26-v2",
        "spe_mm_dino_v1": "spe-mm-dino-v1",
        "spe_mm_dino_v2": "spe-mm-dino-v2",
        #"spe_mm_jepa_wospecenc": "SPE-MM-JEPA wo-SpecEnc"
        "s2_dino_hieraB_4stage_patchs3": "S2 DINO Hiera-B (4stage/patchsize3)",
        "s2_dino_hieraL_3stage_patchs6": "S2 DINO Hiera-L (3stage/patchsize6)",
        "s2_dino_hieraL_4stage_patchs3": "S2 DINO Hiera-L (4stage/patchsize3)",
        "hierab_mm_dino_3stages": "Hiera-B MM DINO (3stages)",
        "hierab_mm_dino_3stages_S2S1": "Hiera-B MM DINO (3stages/S2S1)",
        "hierab_mm_dino_3stages_rescales2": "Hiera-B MM DINO (3stages/rescales2)",
        "hierab_mm_dino_3stages_rescales4": "Hiera-B MM DINO (3stages/rescales4)",
        "hierab_mm_dino_3stages_rescales421": "Hiera-B MM DINO (3stages/rescales421)",
        "hierab_mm_dinov2_3stages": "Hiera-B MM DINOv2 (3stages)",
        "hierab_mm_dinov2_3stages_S2S1": "Hiera-B MM DINOv2 (3stages/S2S1)",
        "hierab_s2_jepa_3stages": "S2 Hiera-B JEPA (3stages)",
        "hierab_s2_jepa_3stages_lossweighting": "S2 Hiera-B JEPA (3stages, lossweight)",
        "s2_dino_hiera_lowbanddrop": "S2 Hiera DINO (low b-drop)",
        "s2_dino_hiera_mediumbanddrop": "S2 Hiera DINO (medium b-drop)",
        "s2_dino_hiera_highbanddrop": "S2 Hiera DINO (high b-drop)",
        "hierab_s2_jepa_lowbanddrop": "S2 Hiera-B JEPA (low b-drop)",
        "hierab_s2_jepa_mediumbanddrop": "S2 Hiera-B JEPA (medium b-drop)",
        "hierab_s2_jepa_lowlambdadetach": "S2 Hiera-B JEPA (low lambda detach)",
        "hierab_s2_jepa_lowlambda": "S2 Hiera-B JEPA (low lambda)",
        "hierab_s2_jepa_verylowlambda": "S2 Hiera-B JEPA (very low lambda)",
        "hierab_s2_jepa_highlambda": "S2 Hiera-B JEPA (high lambda)",
        "hierab_s2_jepa_highlambdadetach": "S2 Hiera-B JEPA (high lambda detach)",
        "hierab_mm_jepa_highlambda": "Hiera-B MM JEPA (high lambda)",
        "hierab_mm_jepa_highlambdadetach": "Hiera-B MM JEPA (high lambda detach)",
        "hierab_mm_jepa_lowlambdadetach": "Hiera-B MM JEPA (low lambda detach)",
    }
    pivot_table = pivot_table.rename(index=rename_mapping)

    # Rename encoder_name to Model (currently "index", not)
    pivot_table = pivot_table.reset_index().rename(columns={'encoder_name': 'Model'})

    # Read pangaea results csv (including TerraMind and Thor)
    df_pangaea = pd.read_csv(f'scripts/analyze_experiments/pangaea_results_{limited_label_fraction}.csv', sep=' ')
    # Drop multiple columns (datasets that could not be reproduced)
    df_pangaea = df_pangaea.drop(['FBP', 'CropMap'], axis=1)

    # Now merge our results df with the original Pangaea df
    merged_df = pd.concat([df_pangaea, pivot_table], ignore_index=True)

    # Calculate the Average Rank:
    # Select only the metric columns (excluding "Model")
    metric_columns = merged_df.columns[1:]  # All except first column ("Model")

    # Convert metrics to numeric, handling any non-numeric entries
    merged_df[metric_columns] = merged_df[metric_columns].apply(pd.to_numeric, errors='coerce')

    # Compute rank for each dataset (descending: higher mIoU = better rank = lower number)
    ranks_df = merged_df[metric_columns].rank(ascending=False, method='min')

    # Add average rank
    merged_df['Average Rank'] = ranks_df.mean(axis=1).round(2)
    
    # Display the result
    print(merged_df)

    latex_table = merged_df.to_latex(
        caption=f"Extended Pangaea results with {int(limited_label_fraction*100)}\% training data in mIoU",
        label="tab:pangaea_benchmark",
        float_format='%.2f',
        index=False,  # Include row indices
    )
    print(latex_table)
