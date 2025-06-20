import os
import math
import torch
import molgrid
import numpy as np
from glob import glob
from sklearn.metrics import roc_auc_score
import pandas as pd

from functions import *
from gnina_dense_model import Dense

# === CONFIGURATION === #
DATA_ROOT = 'DUDE_data'
DATA_ROOT2 = 'dude_vs'
OUTPUT_ROOT = 'ligands_sdf'
OUTPUT_ROOT2 = 'results_DUD_E'
WEIGHTS_PATH = './weights/dense.pt'
TYPES_FILENAME = 'molgrid_input.types'
BATCH_SIZE = 1
NUM_CONFORMERS = math.inf
TOP_N = 3
METHOD = 'max_aff'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure reproducible behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === GLOBALS === #
model = None
target_metrics = {}     # Will store per-target performance metrics
per_target_data = {}    # Will collect per-target predictions

def load_model(input_dims):
    """
    Initialize and load Dense model with pretrained weights.

    Args:
        input_dims (tuple): Expected input dimensions for model.

    Returns:
        torch.nn.Module: Loaded model in evaluation mode.
    """
    mdl = Dense(input_dims).to(DEVICE)
    mdl.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    mdl.eval()
    return mdl

def prepare_gridmaker_and_tensor(provider):
    """
    Set up molgrid GridMaker and an empty tensor placeholder.

    Args:
        provider (molgrid.ExampleProvider): provider for ligand examples.

    Returns:
        grid_maker, dims, tensor: For populating and running model inference.
    """
    gm = molgrid.GridMaker(resolution=0.5, dimension=23.5)
    dims = gm.grid_dimensions(provider.num_types())
    tensor = torch.empty((BATCH_SIZE, *dims), dtype=torch.float32, device=DEVICE)
    return gm, dims, tensor

def process_target(target_folder):
    """
    Handle one target: extract docked ligands, run model inference,
    and compute performance metrics.

    Args:
        target_folder (str): Name of target directory under DATA_ROOT.
    """
    # Construct paths
    tgt1 = os.path.join(DATA_ROOT, target_folder)
    tgt2 = os.path.join(DATA_ROOT2, target_folder)
    receptor = os.path.join(tgt2, 'receptor.pdbqt')
    types_file = os.path.join(tgt1, TYPES_FILENAME)
    out_dir = os.path.join(OUTPUT_ROOT, target_folder)

    # Skip if receptor is missing
    if not os.path.isfile(receptor):
        print(f"[SKIP] Receptor not found for {target_folder}")
        return

    os.makedirs(out_dir, exist_ok=True)

    # Step 1. Unzip any compressed SDFs
    print(f"[INFO] Extracting SDFs for {target_folder}")
    extract_sdf_gz_files(tgt1, "_docked_vina.sdf.gz")
    extract_sdf_gz_files(tgt2, "_docked_vina.sdf.gz")

    # Step 2. Write .types input
    with open(types_file, 'w') as tf:
        # Actives
        process_molecules(os.path.join(tgt1, 'actives_final_docked_vina.sdf'),
                          NUM_CONFORMERS, 1, 'active', out_dir, receptor, tf, 0)
        missing_act = os.path.join(tgt2, 'missing_actives_docked_vina.sdf')
        if os.path.isfile(missing_act) and os.path.getsize(missing_act) > 0:
            process_molecules(missing_act, NUM_CONFORMERS, 1,
                              'active_missing', out_dir, receptor, tf, 0)
        # Decoys
        for idx, decoy in enumerate(glob(os.path.join(tgt1, 'decoys_final_*_docked_vina.sdf'))):
            process_molecules(decoy, NUM_CONFORMERS, 0,
                              'decoy', out_dir, receptor, tf, idx)
        missing_decoys = glob(os.path.join(tgt2, 'missing_decoys_docked_vina.sdf'))
        for missing in missing_decoys:
            if os.path.isfile(missing) and os.path.getsize(missing) > 0:
                process_molecules(missing, NUM_CONFORMERS, 0,
                                  'decoy_missing', out_dir, receptor, tf, idx)

    # Step 3. Initialize molgrid provider
    print(f"[INFO] Populating MolGrid for {target_folder}")
    provider = molgrid.ExampleProvider(data_root='.', balanced=False,
                                       shuffle=False, cache_structs=True)
    provider.populate(types_file)

    # Step 4. Set up grid maker and model
    gm, dims, tensor = prepare_gridmaker_and_tensor(provider)
    global model
    if model is None:
        model = load_model(dims)

    # Prepare tensors for labels and codes
    label_tensor = torch.empty(BATCH_SIZE, dtype=torch.float32, device=DEVICE)
    code_tensor = torch.empty(BATCH_SIZE, dtype=torch.float32, device=DEVICE)

    predictions = {}
    num_batches = math.ceil(provider.size() / BATCH_SIZE)
    print(f"[INFO] Running {num_batches} batches for {target_folder}")

    for _ in range(num_batches):
        batch = provider.next_batch(BATCH_SIZE)
        if batch is None:
            break

        batch.extract_label(0, label_tensor)
        batch.extract_label(1, code_tensor)
        gm.forward(batch, tensor, random_rotation=False, random_translation=0.)
        with torch.no_grad():
            pose, affinity = model(tensor)

        labels = label_tensor.cpu().numpy()
        codes = code_tensor.cpu().numpy().astype(np.int64)
        poses = pose.cpu().numpy()
        affinities = affinity[:, 0].cpu().numpy()

        # Store predictions per code
        for i, code in enumerate(codes):
            entry = predictions.setdefault(code, {
                'labels': [], 'pose_scores': [], 'affinity_scores': []
            })
            entry['labels'].append(labels[i])
            entry['pose_scores'].append(poses[i])
            entry['affinity_scores'].append(affinities[i])

    print(f"[INFO] Computing metrics for {target_folder}")
    compute_metrics(target_folder, predictions)

def compute_metrics(target, predictions):
    """
    Compute ROC AUC, correlations, and early enrichment metrics.

    Populates target_metrics[target] with computed values.
    """
    affs, labs, poses = [], [], []
    for code in sorted(predictions):
        l, p, a = get_data(code, predictions, METHOD, TOP_N)
        labs.append(l); poses.append(p); affs.append(a)

    # Check if both classes exist
    if len(set(labs)) > 1:
        auc_pose = roc_auc_score(labs, poses)
        auc_aff = roc_auc_score(labs, affs)
        boot_aff = bootstrap_roc_auc(labs, affs)
        boot_pose = bootstrap_roc_auc(labs, poses)
        mean_aff, mean_pose = np.mean(boot_aff), np.mean(boot_pose)
        median_aff, median_pose = np.median(boot_aff), np.median(boot_pose)
        ci_aff = (np.percentile(boot_aff, 2.5), np.percentile(boot_aff, 97.5))
        ci_pose = (np.percentile(boot_pose, 2.5), np.percentile(boot_pose, 97.5))
    else:
        auc_pose = auc_aff = None
        mean_aff = median_aff = mean_pose = median_pose = None
        ci_aff = ci_pose = (None, None)

    total_ligs = sum(len(v['affinity_scores']) for v in predictions.values())
    unique_ligs = len({tuple(v['affinity_scores']) for v in predictions.values()})
    n_act = int(sum(labs))
    n_decoy = unique_ligs - n_act

    target_metrics[target] = {
        'labels': labs,
        'pose_scores': poses,
        'affinity_scores': affs,
        'num_ligands': total_ligs,
        'num_unique_ligands': unique_ligs,
        'num_actives': n_act,
        'num_decoys': n_decoy,
        'roc_auc(pose)': auc_pose,
        'roc_auc(affinity)': auc_aff,
        'boot_aff_mean': mean_aff,
        'boot_aff_median': median_aff,
        'boot_aff_ci': ci_aff,
        'boot_pose_mean': mean_pose,
        'boot_pose_median': median_pose,
        'boot_pose_ci': ci_pose,
    }

    pose_efs, pose_nefs = bootstrap_ef_nef(labs, poses, fpr_levels=[0.01, 0.05, 0.1])
    aff_efs, aff_nefs = bootstrap_ef_nef(labs, affs, fpr_levels=[0.01, 0.05, 0.1])

    def summarize(results, prefix):
        for fpr, arr in results.items():
            m = np.mean(arr)
            ci = (np.percentile(arr, 2.5), np.percentile(arr, 97.5))
            target_metrics[target][f'{prefix} {fpr*100:.1f}%'] = m
            target_metrics[target][f'{prefix} {fpr*100:.1f}% CI'] = ci

    summarize(pose_efs, "EF (Pose)")
    summarize(pose_nefs, "NEF (Pose)")
    summarize(aff_efs, "EF (Affinity)")
    summarize(aff_nefs, "NEF (Affinity)")

# === MAIN EXECUTION === #
for folder in sorted(os.listdir(DATA_ROOT)):
    path = os.path.join(DATA_ROOT, folder)
    if os.path.isdir(path):
        print(f"\n[INFO] Processing target: {folder}")
        process_target(folder)

def summarize_metric(key):
    vals = [m[key] for m in target_metrics.values() if key in m and m[key] is not None]
    return np.mean(vals), np.median(vals)

print("\n==== AGGREGATE METRICS ====")
for key in [
    'roc_auc(pose)', 'roc_auc(affinity)',
    'EF (Pose) 1.0%', 'EF (Affinity) 1.0%',
    'NEF (Pose) 1.0%', 'NEF (Affinity) 1.0%'
]:
    mn, md = summarize_metric(key)
    print(f"{key} -> Mean: {mn:.4f}, Median: {md:.4f}")

os.makedirs(OUTPUT_ROOT2, exist_ok=True)

# Save all target metrics to CSV
rows = {t: {k: str(v) for k, v in m.items() if k not in ('labels','pose_scores','affinity_scores')}
        for t, m in target_metrics.items()}
pd.DataFrame.from_dict(rows, orient='index') \
  .to_csv(os.path.join(OUTPUT_ROOT2, "full_target_metrics.csv"))

# === PLOTS === #
print("\n==== GENERATING FIGURES ====")
plot_bootstrapped_metrics(target_metrics, save_path='metrics_affinity.png', mode='Affinity',
                          auc_mode='aff', reference_file='reference_metrics.xlsx')
plot_bootstrapped_metrics(target_metrics, save_path='metrics_pose.png', mode='Pose',
                          auc_mode='pose', reference_file='reference_metrics.xlsx')
plot_ef_nef_grouped_bar_with_ci(target_metrics, mode='Affinity')
plot_ef_nef_grouped_bar_with_ci(target_metrics, mode='Pose')
plot_roc_and_distributions(target_metrics)
