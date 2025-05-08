import os
import math
import torch
import molgrid
import numpy as np
from glob import glob
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib import pyplot as plt
import seaborn as sns
from functions import extract_sdf_gz_files, process_actives, process_decoys
from gnina_dense_model import Dense
from openbabel import openbabel
openbabel.OBMessageHandler().SetOutputLevel(0)  # Suppress Open Babel warnings
openbabel.obErrorLog.SetOutputLevel(0)  # Suppress Open Babel warnings

# === CONFIGURATION SECTION === #
DATA_ROOT = 'data'
OUTPUT_ROOT = 'ligands_sdf'
WEIGHTS_PATH = './weights/dense.pt'
TYPES_FILENAME = 'molgrid_input.types'
RECEPTOR_BASE = './DUD_E_withoutfgfr1'
BATCH_SIZE = 1
num_conformers = 10  # Number of conformers to process
top_n = 3  # Number of top entries to consider for mean calculation
method = 'max_aff'  # Method to select data from predictions
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure deterministic results from cuDNN (may reduce speed slightly, but ensures reproducibility)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === GLOBAL VARIABLES === #
model = None  # Loaded once and reused
target_metrics = {}  # Store ROC AUC and correlation per target
per_target_data = {}  # Store scores for visualization

# === UTILITY FUNCTIONS === #

def load_model(input_dims):
    """Load the Dense model only once with pretrained weights."""
    m = Dense(input_dims).to(DEVICE)
    m.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    m.eval()
    return m

def prepare_gridmaker_and_tensor(provider):
    """Set up grid maker and allocate input tensor based on example provider type info."""
    grid_maker = molgrid.GridMaker(resolution=0.5, dimension=23.5)
    dims = grid_maker.grid_dimensions(provider.num_types())
    tensor_shape = (BATCH_SIZE,) + tuple(dims)
    tensor = torch.empty(tensor_shape, dtype=torch.float32, device=DEVICE)
    return grid_maker, dims, tensor

def process_target(target_folder):
    """Process a single target directory and perform inference."""
    target_path = os.path.join(DATA_ROOT, target_folder)
    receptor_path = os.path.join(RECEPTOR_BASE, target_folder, 'receptor.pdb')
    types_file = os.path.join(target_path, TYPES_FILENAME)
    output_dir = os.path.join(OUTPUT_ROOT, target_folder)

    # Skip if receptor file doesn't exist
    if not os.path.isfile(receptor_path):
        print(f"Missing receptor for {target_folder}, skipping...")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Extract SDF files (if gzipped)
    extract_sdf_gz_files(target_path)

    # Step 2: Write actives and decoys to types file
    with open(types_file, 'w') as tf:
        process_actives(
            sdf_path=os.path.join(target_path, 'actives_final_docked_vina.sdf'),
            number=num_conformers, label=1, prefix='active',
            output_dir=output_dir, receptor_path=receptor_path,
            types_file_handle=tf
        )

        # Process all decoy batches
        for batch_num, decoy_file in enumerate(sorted(glob(os.path.join(target_path, 'decoys_final_*_docked_vina.sdf')))):
            print(f"Processing decoy batch {batch_num}: {decoy_file}")
            process_decoys(
                sdf_path=decoy_file, number=num_conformers, label=0,
                prefix='decoy', output_dir=output_dir,
                receptor_path=receptor_path, types_file_handle=tf,
                batch_num=batch_num
            )

    # Step 3: Setup MolGrid provider and tensors
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
                predictions[code] = {'labels': [], 'pose_scores': [], 'affinity_scores': []}
            predictions[code]['labels'].append(labels_np[i])
            predictions[code]['pose_scores'].append(poses_np[i])
            predictions[code]['affinity_scores'].append(affinities_np[i])

    # Compute metrics for the current target
    compute_metrics(target_folder, predictions)

def get_data(code, predictions, method, top_n):
    """
    Extract data for a given input code from predictions based on the specified method.

    Parameters:
    - code (str): Identifier for the prediction entry.
    - predictions (dict): Dictionary containing prediction data. Each entry should have
                          'labels', 'pose_scores', and 'affinity_scores' as NumPy arrays.
    - method (str): One of the following:
        - 'max_aff': Selects data from the entry with the highest affinity score.
        - 'max_pose': Selects data from the entry with the highest pose score.
        - 'mean': Computes the mean of labels, pose scores, and affinity scores from the top
                  3 entries with the highest affinity scores.

    Returns:
    - label (float): Selected or averaged label value.
    - pose (float): Selected or averaged pose score.
    - affinity (float): Selected or averaged affinity score.
    """
    entry = predictions[code]

    if method == 'max_aff':
        # Use the entry with the highest affinity score
        idx = np.argmax(entry['affinity_scores'])
        label = entry['labels'][idx]
        pose = entry['pose_scores'][idx]
        affinity = entry['affinity_scores'][idx]

    elif method == 'max_pose':
        # Use the entry with the highest pose score
        idx = np.argmax(entry['pose_scores'])
        label = entry['labels'][idx]
        pose = entry['pose_scores'][idx]
        affinity = entry['affinity_scores'][idx]

    elif method == 'mean':
        # Get indices of the top 3 entries by affinity score (sorted descending)
        top_idxs = np.argsort(entry['affinity_scores'])[::-1][:top_n]

        # Compute mean values from the top 3 aligned entries
        label = np.mean(entry['labels'][top_idxs])
        pose = np.mean(entry['pose_scores'][top_idxs])
        affinity = np.mean(entry['affinity_scores'][top_idxs])

    else:
        raise ValueError("Method must be 'max_aff', 'max_pose', or 'mean'.")

    return label, pose, affinity

def compute_nef_1_percent(all_affinities, all_labels):
    """
    Compute the Normalized Enrichment Factor (NEF) at 1% and Enrichment Factor (EF) at 1%.

    Returns:
    - nef_1_percent (float): Normalized EF at 1%
    - ef_1_percent (float): Raw Enrichment Factor at 1%
    """
    total = len(all_labels) # total number of ligands
    num_actives = sum(all_labels) # total number of actives
    hit_rate = num_actives / total if total > 0 else 0 # total fraction of actives
    print(total, num_actives, hit_rate)

    top_n = max(1, total // 100)  # 1% of all ligands, ensure at least one element
    top_indices = np.argsort(all_affinities)[::-1][:top_n] # indices of top 1% by affinity
    top_actives = sum(all_labels[i] for i in top_indices) # count of actives in top 1%
    ef_1_percent = top_actives / top_n # raw enrichment factor at 1%
    nef_1_percent = ef_1_percent / hit_rate if hit_rate > 0 else 0
    print(top_n, top_indices, top_actives)

    return nef_1_percent, ef_1_percent

def compute_metrics(target, predictions):
    """Compute ROC AUC, Pearson correlation, and store max-score data."""
    all_affinities, all_labels, all_poses = [], [], []

    for code in sorted(predictions.keys()):
        label, pose, affinity = get_data(code, predictions, method=method, top_n=top_n)
        
        all_affinities.append(affinity)
        all_labels.append(label)
        all_poses.append(pose)

    # Compute metrics only if both classes are present
    has_both_classes = len(set(all_labels)) > 1
    roc_auc = roc_auc_score(all_labels, all_poses) if has_both_classes else None
    pearson_corr = pearsonr(all_poses, all_affinities)[0] if has_both_classes else None
    nef_1_percent, ef_1_percent = compute_nef_1_percent(all_affinities, all_labels)

    # Store metrics and data
    target_metrics[target] = {
        'num_ligands': len(predictions),
        'roc_auc': roc_auc,
        'pearson_correlation': pearson_corr,
        'nef_1_percent': nef_1_percent,
        'ef_1_percent': ef_1_percent
    }
    per_target_data[target] = {
        'labels': all_labels,
        'pose_scores': all_poses,
        'affinity_scores': all_affinities
    }

# === MAIN EXECUTION LOOP === #
for folder in sorted(os.listdir(DATA_ROOT)):
    if os.path.isdir(os.path.join(DATA_ROOT, folder)):
        print(f"\nProcessing target: {folder}")
        process_target(folder)

print("\n==== Summary of Metrics by Target ====")
for target, metrics in target_metrics.items():
    print(f"\nTarget: {target}")
    for key, value in metrics.items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: N/A")

# === VISUALIZATION FUNCTIONS === #

def plot_results():
    """Plot ROC curves and affinity distribution histograms per target."""
    num_targets = len(per_target_data)
    cols, rows = 3, (num_targets + 2) // 3
    fig_roc, axs_roc = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    fig_dist, axs_dist = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axs_roc, axs_dist = axs_roc.flatten(), axs_dist.flatten()

    for idx, (target, data) in enumerate(per_target_data.items()):
        labels, poses, affinities = data['labels'], data['pose_scores'], data['affinity_scores']

        # ROC Curve
        ax_roc = axs_roc[idx]
        if len(set(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, poses)
            auc = roc_auc_score(labels, poses)
            ax_roc.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color='blue')
        else:
            ax_roc.text(0.5, 0.5, "Insufficient label variation", ha='center', va='center')

        ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax_roc.set_title(target)
        ax_roc.set_xlabel('FPR')
        ax_roc.set_ylabel('TPR')
        ax_roc.legend()
        ax_roc.grid(True)

        # Affinity Distribution
        ax_dist = axs_dist[idx]
        actives = [a for a, l in zip(affinities, labels) if l == 1]
        decoys = [a for a, l in zip(affinities, labels) if l == 0]
        sns.histplot(actives, kde=True, color='green', label='Actives', stat="density", ax=ax_dist, bins=20)
        sns.histplot(decoys, kde=True, color='red', label='Decoys', stat="density", ax=ax_dist, bins=20)
        ax_dist.set_title(target)
        ax_dist.set_xlabel('Predicted Affinity')
        ax_dist.set_ylabel('Density')
        ax_dist.legend()
        ax_dist.grid(True)

    # Remove unused subplots
    for ax in axs_roc[num_targets:]: fig_roc.delaxes(ax)
    for ax in axs_dist[num_targets:]: fig_dist.delaxes(ax)

    fig_roc.tight_layout()
    fig_dist.tight_layout()
    fig_roc.savefig("roc_curves.png", dpi=300)
    fig_dist.savefig("affinity_distributions.png", dpi=300)
    print("\nSaved ROC curves to: roc_curves.png")
    print("Saved affinity distributions to: affinity_distributions.png")

    """Plot ROC curves and affinity distribution histograms per target."""
    num_targets = len(per_target_data)
    cols, rows = 3, (num_targets + 2) // 3
    fig_roc, axs_roc = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    fig_dist, axs_dist = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axs_roc, axs_dist = axs_roc.flatten(), axs_dist.flatten()

    for idx, (target, data) in enumerate(per_target_data.items()):
        labels, poses, affinities = data['labels'], data['pose_scores'], data['affinity_scores']

        # ROC Curve
        ax_roc = axs_roc[idx]
        if len(set(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, poses)
            auc = roc_auc_score(labels, poses)
            ax_roc.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color='blue')
        else:
            ax_roc.text(0.5, 0.5, "Insufficient label variation", ha='center', va='center')

        ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax_roc.set_title(target)
        ax_roc.set_xlabel('FPR')
        ax_roc.set_ylabel('TPR')
        ax_roc.legend()
        ax_roc.grid(True)

        # Affinity Distribution
        ax_dist = axs_dist[idx]
        actives = [a for a, l in zip(affinities, labels) if l == 1]
        decoys = [a for a, l in zip(affinities, labels) if l == 0]
        sns.histplot(actives, kde=True, color='green', label='Actives', stat="density", ax=ax_dist, bins=20)
        sns.histplot(decoys, kde=True, color='red', label='Decoys', stat="density", ax=ax_dist, bins=20)
        ax_dist.set_title(target)
        ax_dist.set_xlabel('Predicted Affinity')
        ax_dist.set_ylabel('Density')
        ax_dist.legend()
        ax_dist.grid(True)

    # Remove unused subplots
    for ax in axs_roc[num_targets:]: fig_roc.delaxes(ax)
    for ax in axs_dist[num_targets:]: fig_dist.delaxes(ax)

    fig_roc.tight_layout()
    fig_dist.tight_layout()
    fig_roc.savefig("roc_curves.png", dpi=300)
    fig_dist.savefig("affinity_distributions.png", dpi=300)
    print("\nSaved ROC curves to: roc_curves.png")
    print("Saved affinity distributions to: affinity_distributions.png")

# Generate plots
plot_results()

def plot_nef_and_auc_scatter(target_metrics):

    targets = list(target_metrics.keys())

    # Extract metrics from the dictionary
    nef_scores = [target_metrics[t]['nef_1_percent'] if 'nef_1_percent' in target_metrics[t] else np.nan for t in targets]
    ef_scores = [target_metrics[t]['ef_1_percent'] if 'ef_1_percent' in target_metrics[t] else np.nan for t in targets]
    auc_scores = [target_metrics[t]['roc_auc'] if target_metrics[t]['roc_auc'] is not None else np.nan for t in targets]

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(12, 7), constrained_layout=True)

    # NEF 1%
    axs[0].scatter(targets, nef_scores, color='tab:blue', s=18)
    axs[0].set_ylabel('NEF1%')
    axs[0].set_title('NEF1% per Target')
    axs[0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    axs[0].grid(True)

    # EF 1%
    axs[1].scatter(targets, ef_scores, color='tab:green', s=18)
    axs[1].set_ylabel('EF1%')
    axs[1].set_title('EF1% per Target')
    axs[1].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    axs[1].grid(True)

    # ROC AUC
    axs[2].scatter(targets, auc_scores, color='tab:orange', s=18)
    axs[2].set_ylabel('ROC AUC')
    axs[2].set_title('ROC AUC per Target')
    axs[2].set_xticks(range(len(targets)))
    axs[2].set_xticklabels(targets, rotation=90, fontsize=8)
    axs[2].grid(True)

    plt.savefig("combined_metrics_scatter.png", dpi=300)
    print("Saved combined metrics plot to: combined_metrics_scatter.png")
plot_nef_and_auc_scatter(target_metrics)
