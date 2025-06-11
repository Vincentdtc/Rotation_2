# === IMPORTS === #
import os
import math
import torch
import molgrid
import numpy as np
from glob import glob
from sklearn.metrics import roc_auc_score
import pandas as pd

# Local modules
from functions_vec import *
from gnina_dense_model_vec import Dense

# === CONFIG === #
DATA_ROOT, DATA_ROOT2, OUTPUT_ROOT, OUTPUT_ROOT2 = 'DUDE_data', 'dude_vs', 'ligands_sdf', 'results_DUD_E'
WEIGHTS_PATH, TYPES_FILENAME = './weights/dense.pt', 'molgrid_input.types'
BATCH_SIZE, NUM_CONFORMERS, TOP_N, METHOD = 1, math.inf, 3, 'max_aff'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set deterministic mode for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === GLOBALS === #
model = None
target_metrics = {}     # For storing per-target ROC AUC and correlations
per_target_data = {}    # For storing per-target predictions

# === MODEL LOADING === #
def load_model(input_dims):
    model = Dense(input_dims).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    return model

# === GRID SETUP === #
def prepare_gridmaker_and_tensor(provider):
    grid_maker = molgrid.GridMaker(resolution=0.5, dimension=23.5)
    dims = grid_maker.grid_dimensions(provider.num_types())
    tensor = torch.empty((BATCH_SIZE, *dims), dtype=torch.float32, device=DEVICE)
    return grid_maker, dims, tensor

# === TARGET PROCESSING === #
def process_target(target_folder):
    """Process one target for active/decoy inference and feature generation."""
    target_path = os.path.join(DATA_ROOT, target_folder)
    target_path_missing = os.path.join(DATA_ROOT2, target_folder)
    receptor_file = os.path.join(DATA_ROOT2, target_folder, 'receptor.pdbqt')
    types_file = os.path.join(target_path, TYPES_FILENAME)
    output_dir = os.path.join(OUTPUT_ROOT, target_folder)

    # Ensure receptor exists
    if not os.path.isfile(receptor_file):
        print(f"[SKIP] Receptor not found for {target_folder}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Unzip any compressed SDFs
    print(f"[INFO] Extracting SDF files for target: {target_folder}")
    extract_sdf_gz_files(target_path, "_docked_vina.sdf.gz")
    extract_sdf_gz_files(target_path_missing, "_docked_vina.sdf.gz")

    # Step 2: Process actives and decoys into .types format
    with open(types_file, 'w') as tf:
        # === Actives === #
        actives_sdf = os.path.join(target_path, 'actives_final_docked_vina.sdf')
        actives_missing_sdf = os.path.join(target_path_missing, 'missing_actives_docked_vina.sdf')
        
        process_molecules(
                sdf_path=actives_sdf,
                number=NUM_CONFORMERS,
                label=1,
                prefix='active',
                output_dir=output_dir,
                receptor_path=receptor_file,
                types_file_handle=tf,
                batch_num=0
            )

        # Check if there are missing actives in dataroot2
        if os.path.isfile(actives_missing_sdf) and os.path.getsize(actives_missing_sdf) > 0:
            process_molecules(
                sdf_path=actives_missing_sdf,
                number=NUM_CONFORMERS,
                label=1,
                prefix='active_missing',
                output_dir=output_dir,
                receptor_path=receptor_file,
                types_file_handle=tf,
                batch_num=0
            )

        # === Decoys === #
        decoy_files = glob(os.path.join(target_path, 'decoys_final_*_docked_vina.sdf'))
        missing_decoys = glob(os.path.join(target_path_missing, 'missing_decoys_docked_vina.sdf'))

        for batch_num, decoy_file in enumerate(decoy_files):
            decoy_batch_dir = os.path.join(output_dir, f"decoy_batch_{batch_num}")
            os.makedirs(decoy_batch_dir, exist_ok=True)
            
            process_molecules(
                    sdf_path=decoy_file,
                    number=NUM_CONFORMERS,
                    label=0,
                    prefix='decoy',
                    output_dir=output_dir,
                    receptor_path=receptor_file,
                    types_file_handle=tf,
                    batch_num=batch_num
                )
        #Process missing decoys        
        for missing_file in missing_decoys:
            if os.path.isfile(missing_file) and os.path.getsize(missing_file) > 0:
                process_molecules(
                    sdf_path=missing_file,
                    number=NUM_CONFORMERS,
                    label=0,
                    prefix='decoy_missing',
                    output_dir=output_dir,
                    receptor_path=receptor_file,
                    types_file_handle=tf,
                    batch_num=batch_num
                    )

    # Step 3: Setup MolGrid provider and tensors
    print(f"[INFO] Setting up MolGrid provider for target: {target_folder}")
    provider = molgrid.ExampleProvider(data_root='.', balanced=False, shuffle=False, cache_structs=True)
    provider.populate(types_file)

    grid_maker, dims, tensor = prepare_gridmaker_and_tensor(provider)

    # Step 4: Initialize model once
    global model
    if model is None:
        model = load_model(dims)

    # Allocate label and code tensors
    float_labels = torch.empty(BATCH_SIZE, dtype=torch.float32, device=DEVICE)
    float_codes = torch.empty(BATCH_SIZE, dtype=torch.float32, device=DEVICE)

    predictions = {}
    num_batches = math.ceil(provider.size() / BATCH_SIZE)

    print(f"[INFO] Processing {num_batches} batches for target: {target_folder}")
    for _ in range(num_batches):
        batch = provider.next_batch(BATCH_SIZE)
        if batch is None:
            break

        # Extract labels and ligand codes (used as unique IDs)
        batch.extract_label(0, float_labels)
        batch.extract_label(1, float_codes)

        # Forward pass: create grid and perform inference
        grid_maker.forward(batch, tensor, random_rotation=False, random_translation=0.0)
        
        with torch.no_grad():
            pose, affinity = model(tensor)

        # Convert outputs to NumPy arrays once
        labels_np = float_labels.cpu().numpy()
        codes_np = float_codes.cpu().numpy().astype(np.int64)
        poses_np = pose.cpu().numpy()
        affinities_np = affinity[:, 0].cpu().numpy()

        # Batch insert predictions
        for i, code in enumerate(codes_np):
            if code not in predictions:
                predictions[code] = {
                    'labels': [], 
                    'pose_scores': [], 
                    'affinity_scores': [],
                    }
            predictions[code]['labels'].append(labels_np[i])
            predictions[code]['pose_scores'].append(poses_np[i])
            predictions[code]['affinity_scores'].append(affinities_np[i])

    # Compute metrics for the current target
    print(f"[INFO] Computing metrics for target: {target_folder}")
    compute_metrics(target_folder, predictions)

def compute_metrics(target, predictions):
    """Compute ROC AUC, Pearson correlation, and store max-score data."""
    all_affinities, all_labels, all_poses= [], [], []

    # Helper function to extract data from predictions
    for code in sorted(predictions.keys()):
        label, pose, affinity = get_data(code, predictions, METHOD, TOP_N)
        all_affinities.append(affinity)
        all_labels.append(label)
        all_poses.append(pose)

    # Compute metrics only if both classes are present
    has_both_classes = len(set(all_labels)) > 1

    if has_both_classes:
        roc_auc_pose = roc_auc_score(all_labels, all_poses)
        roc_auc_aff = roc_auc_score(all_labels, all_affinities)

        # Bootstrap AUCs
        boot_aff = bootstrap_roc_auc(all_labels, all_affinities)
        boot_pose = bootstrap_roc_auc(all_labels, all_poses)

        # Compute central tendency
        mean_aff = np.mean(boot_aff)
        mean_pose = np.mean(boot_pose)
        median_aff = np.median(boot_aff)
        median_pose = np.median(boot_pose)

        # Compute confidence intervals
        ci_aff = (np.percentile(boot_aff, 2.5), np.percentile(boot_aff, 97.5))
        ci_pose = (np.percentile(boot_pose, 2.5), np.percentile(boot_pose, 97.5))
    else:
        roc_auc_pose = roc_auc_aff = None
    
    num_ligands = sum(len(v) for v in predictions.values())
    num_unique_ligands = len(set(all_affinities))
    num_unique_actives = int(sum(all_labels))
    num_unique_decoys = num_unique_ligands - num_unique_actives

    # Store metrics and data
    target_metrics[target] = {
        'labels': all_labels,
        'pose_scores': all_poses,
        'affinity_scores': all_affinities,
        'num_ligands': num_ligands,
        'num_unique_ligands': num_unique_ligands,
        'num_actives': num_unique_actives,
        'num_decoys': num_unique_decoys,
        'roc_auc(pose)': roc_auc_pose,
        'roc_auc(affinity)': roc_auc_aff,
        'boot_aff_mean': mean_aff,
        'boot_aff_median': median_aff,
        'boot_aff_ci': ci_aff,
        'boot_pose_mean': mean_pose,
        'boot_pose_median': median_pose,
        'boot_pose_ci': ci_pose,
    }

    # Bootstrap enrichment factors
    boot_efs_pose, boot_nefs_pose = bootstrap_ef_nef(all_labels, all_poses, fpr_levels=[0.01, 0.05, 0.1])
    boot_efs_aff, boot_nefs_aff = bootstrap_ef_nef(all_labels, all_affinities, fpr_levels=[0.01, 0.05, 0.1])

    def summarize_bootstrapped_metric(boot_results, prefix, target_metrics_entry):
        for fpr, ef_list in boot_results.items():
            ef_mean = np.mean(ef_list)
            ef_ci = (np.percentile(ef_list, 2.5), np.percentile(ef_list, 97.5))

            target_metrics_entry[f'{prefix} {fpr*100:.1f}%'] = ef_mean
            target_metrics_entry[f'{prefix} {fpr*100:.1f}% CI'] = ef_ci

    summarize_bootstrapped_metric(boot_efs_pose, "EF (Pose)", target_metrics[target])
    summarize_bootstrapped_metric(boot_nefs_pose, "NEF (Pose)", target_metrics[target])
    summarize_bootstrapped_metric(boot_efs_aff, "EF (Affinity)", target_metrics[target])
    summarize_bootstrapped_metric(boot_nefs_aff, "NEF (Affinity)", target_metrics[target])

# === MAIN EXECUTION LOOP === #
for folder in sorted(os.listdir(DATA_ROOT)):
    if os.path.isdir(os.path.join(DATA_ROOT, folder)):
        print(f"\n[INFO] Processing target: {folder}")
        process_target(folder)

# === OVERALL SUMMARY METRICS === #
def summarize_metric(metric_key):
    values = [metrics[metric_key] for metrics in target_metrics.values()
              if metric_key in metrics and metrics[metric_key] is not None]
    return np.mean(values), np.median(values)

# Collect and print summary for each required metric
summary_keys = [
    'roc_auc(pose)', 
    'roc_auc(affinity)',
    'EF (Pose) 1.0%', 
    'EF (Affinity) 1.0%',
    'NEF (Pose) 1.0%', 
    'NEF (Affinity) 1.0%',
]

print("\n==== AGGREGATE METRICS OVER ALL TARGETS ====")
for key in summary_keys:
    mean_val, median_val = summarize_metric(key)
    print(f"{key} -> Mean: {mean_val:.4f}, Median: {median_val:.4f}")

# Ensure the directory exists
os.makedirs(OUTPUT_ROOT2, exist_ok=True)

# Save all metrics to CSV
excluded_keys = {'labels', 'affinity_scores', 'pose_scores'}
stringified_metrics = {}
for target, metrics in target_metrics.items():
    filtered = {k: str(v) for k, v in metrics.items() if k not in excluded_keys}
    stringified_metrics[target] = filtered

df = pd.DataFrame.from_dict(stringified_metrics, orient='index')
df.to_csv(os.path.join(OUTPUT_ROOT2, "full_target_metrics.csv"))

# === VISUALIZATION FUNCTIONS === #
print("\n==== SAVING RESULTS ====")
plot_bootstrapped_metrics(target_metrics, save_path='metrics_affinity.png', mode='Affinity', auc_mode='aff', reference_file='reference_metrics.xlsx')
plot_bootstrapped_metrics(target_metrics, save_path='metrics_pose.png', mode='Pose', auc_mode='pose', reference_file='reference_metrics.xlsx')
plot_ef_nef_grouped_bar_with_ci(target_metrics, mode='Affinity')
plot_ef_nef_grouped_bar_with_ci(target_metrics, mode='Pose')
plot_roc_and_distributions(target_metrics)
