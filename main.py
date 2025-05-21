# === IMPORTS === #
import os
import math
import torch
import molgrid
import numpy as np
from glob import glob
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from openbabel import openbabel

# Local modules
from functions import *
from gnina_dense_model import Dense
from fix_nitrogens import *
from convert_ligs import *

# Suppress Open Babel warnings
openbabel.OBMessageHandler().SetOutputLevel(0)  
openbabel.obErrorLog.SetOutputLevel(0)

# === CONFIGURATION === #
DATA_ROOT = 'DUDE_data'
RECEPTOR_BASE = './DUD_E_withoutfgfr1'
OUTPUT_ROOT = 'ligands_sdf'
OUTPUT_ROOT2 = 'ligs_sdf'
WEIGHTS_PATH = './weights/dense.pt'
TYPES_FILENAME = 'molgrid_input.types'

BATCH_SIZE = 1
NUM_CONFORMERS = 10
TOP_N = 3
METHOD = 'max_aff'

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
    """Load the Dense model with pretrained weights."""
    model = Dense(input_dims).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    return model

# === GRID SETUP === #
def prepare_gridmaker_and_tensor(provider):
    """Create grid maker and allocate input tensor."""
    grid_maker = molgrid.GridMaker(resolution=0.5, dimension=23.5)
    dims = grid_maker.grid_dimensions(provider.num_types())
    tensor_shape = (BATCH_SIZE,) + tuple(dims)
    tensor = torch.empty(tensor_shape, dtype=torch.float32, device=DEVICE)
    return grid_maker, dims, tensor

# === TARGET PROCESSING === #
def process_target(target_folder):
    """Process one target for active/decoy inference and feature generation."""
    target_path = os.path.join(DATA_ROOT, target_folder)
    receptor_file = os.path.join(RECEPTOR_BASE, target_folder, 'receptor.pdb')
    types_file = os.path.join(target_path, TYPES_FILENAME)
    output_dir = os.path.join(OUTPUT_ROOT, target_folder)
    output_dir2 = os.path.join(OUTPUT_ROOT2, target_folder)

    # Ensure receptor exists
    if not os.path.isfile(receptor_file):
        print(f"[SKIP] Receptor not found for {target_folder}")
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)

    # Step 1: Unzip any compressed SDFs
    extract_sdf_gz_files(target_path)

    # Step 2: Process actives and decoys into .types format
    with open(types_file, 'w') as tf:
        print(f"[INFO] Processing target: {target_folder}")

        # === Actives === #
        print(" - Processing actives...")
        actives_sdf = os.path.join(target_path, 'actives_final_docked_vina.sdf')
        processed_actives_sdf = os.path.join(output_dir2, f"{target_folder}_processed.sdf")
        process_sdf(actives_sdf, processed_actives_sdf)
        
        process_molecules(
            sdf_path=processed_actives_sdf,
            number=NUM_CONFORMERS,
            label=1,
            prefix='active',
            output_dir=output_dir,
            receptor_path=receptor_file,
            types_file_handle=tf,
            batch_num=0
        )

        # === Decoys === #
        print(" - Processing decoys...")
        decoy_files = sorted(glob(os.path.join(target_path, 'decoys_final_*_docked_vina.sdf')))

        for batch_num, decoy_file in enumerate(decoy_files):
            print(f"   - Batch {batch_num}: {decoy_file}")
            decoy_batch_dir = os.path.join(output_dir2, f"decoy_batch_{batch_num}")
            os.makedirs(decoy_batch_dir, exist_ok=True)

            processed_decoy_sdf = os.path.join(decoy_batch_dir, f"{target_folder}_decoy_batch_{batch_num}_processed.sdf")
            process_sdf(decoy_file, processed_decoy_sdf)
            process_molecules(
                sdf_path=processed_decoy_sdf,
                number=NUM_CONFORMERS,
                label=0,
                prefix='decoy',
                output_dir=output_dir,
                receptor_path=receptor_file,
                types_file_handle=tf,
                batch_num=batch_num
            )

    # Step 3: Setup MolGrid provider and tensors
    provider = molgrid.ExampleProvider(data_root='.', balanced=False, shuffle=True, cache_structs=True)
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

def compute_metrics(target, predictions):
    """Compute ROC AUC, Pearson correlation, and store max-score data."""
    all_affinities, all_labels, all_poses = [], [], []

    for code in sorted(predictions.keys()):
        label, pose, affinity = get_data(code, predictions, METHOD, TOP_N)
        
        all_affinities.append(affinity)
        all_labels.append(label)
        all_poses.append(pose)

    # Compute metrics only if both classes are present
    has_both_classes = len(set(all_labels)) > 1
    roc_auc_pose = roc_auc_score(all_labels, all_poses) if has_both_classes else None
    roc_auc_aff = roc_auc_score(all_labels, all_affinities) if has_both_classes else None
    pearson_corr = pearsonr(all_poses, all_affinities)[0] if has_both_classes else None
    nef_1, ef_1 = compute_enrichment_factors(all_affinities, all_labels, level=1)
    nef_5, ef_5 = compute_enrichment_factors(all_affinities, all_labels, level=5)
    nef_10, ef_10 = compute_enrichment_factors(all_affinities, all_labels, level=10)
    efs = compute_roc_enrichment_factors(all_labels, all_affinities, fpr_levels=[0.005, 0.01, 0.02, 0.05])

    # Store metrics and data
    target_metrics[target] = {
        'num_ligands': int(len(predictions)),
        'num_unique_ligands': int(len(all_affinities)),
        'num_actives': int(sum(all_labels)),
        'num_decoys': int(len(all_labels) - sum(all_labels)),
        'roc_auc(pose)': roc_auc_pose,
        'roc_auc(affinity)': roc_auc_aff,
        'pearson_correlation': round(pearson_corr,2),
        'NEF 1%': nef_1,
        'EF 1%': ef_1,
        'NEF 5%': nef_5,
        'EF 5%': ef_5,
        'NEF 10%': nef_10,
        'EF 10%': ef_10
    }

    per_target_data[target] = {
        'labels': all_labels,
        'pose_scores': all_poses,
        'affinity_scores': all_affinities
    }

    for level, factor in efs.items():
        target_metrics[target][f'ROC EF {level*100:.1f}%'] = factor

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
plot_results(per_target_data)
plot_ef_nef_grouped_bar(target_metrics)
plot_roc_ef_grouped_bar(target_metrics, save_path='roc_ef_by_target.png')
