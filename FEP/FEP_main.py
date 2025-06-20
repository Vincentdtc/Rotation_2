import os
import math
import torch
import molgrid
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, kendalltau, spearmanr

from FEP_functions import *
from FEP_plotting import *
from gnina_dense_model import Dense
from default2018_model import Net

# === CONFIGURATION === #
DATA_ROOT = 'FEP/FEP_data'                    # Root directory of FEP data
OUTPUT_ROOT = 'FEP/FEP_ligands_sdf'           # Directory to save processed ligand data
WEIGHTS = {
    'Dense': './weights/dense.pt',
    'Def2018': './weights/crossdock_default2018.pt'
}
TYPES_FILENAME = 'molgrid_input.types'        # Filename to store molecular types for molgrid
RECEPTOR_BASE = DATA_ROOT
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make GPU operations deterministic for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load FEP benchmark data
FEP_data = load_fep_data('FEP/FEP_benchmark.csv')
target_metrics = {}  # To store evaluation metrics for each target/model

def load_model(model_type, input_dims):
    """
    Load the specified model with pretrained weights.
    Args:
        model_type (str): Either 'Dense' or 'Def2018'
        input_dims (tuple): Input dimension size for the model
    Returns:
        model (torch.nn.Module): Loaded model set to eval mode on correct device
    """
    if model_type == 'Dense':
        model = Dense(input_dims)
    else:
        model = Net(input_dims)
    model.load_state_dict(torch.load(WEIGHTS[model_type], map_location=DEVICE))
    return model.to(DEVICE).eval()

def prepare_grid(provider):
    """
    Prepare molgrid grid maker and tensor placeholder for batch inputs.
    Args:
        provider (molgrid.ExampleProvider): Data provider for molecules
    Returns:
        grid_maker, dims, tensor: Grid maker object, grid dimensions, and tensor to hold grids
    """
    grid_maker = molgrid.GridMaker(resolution=0.5, dimension=23.5)
    dims = grid_maker.grid_dimensions(provider.num_types())
    tensor = torch.empty((BATCH_SIZE,) + tuple(dims), dtype=torch.float32, device=DEVICE)
    return grid_maker, dims, tensor

def run_model_on_batch(batch, grid_maker, tensor, model):
    """
    Runs the model on a single batch and returns results.
    Args:
        batch: Batch of data from molgrid provider
        grid_maker: molgrid.GridMaker object
        tensor: Preallocated tensor for grid input
        model: Loaded PyTorch model
    Returns:
        labels (np.array): True labels from batch
        codes (np.array): Codes identifying each ligand
        pose (np.array): Pose scores predicted by model
        affinity (np.array): Affinity scores predicted by model
    """
    labels = torch.empty(BATCH_SIZE, dtype=torch.float32, device=DEVICE)
    codes = torch.empty(BATCH_SIZE, dtype=torch.float32, device=DEVICE)
    batch.extract_label(0, labels)  # Extract true labels
    batch.extract_label(1, codes)   # Extract ligand codes
    grid_maker.forward(batch, tensor, random_rotation=False, random_translation=0.0)
    with torch.no_grad():
        pose, affinity = model(tensor)
    return labels.cpu().numpy(), codes.cpu().numpy().astype(int), pose.cpu().numpy(), affinity[:, 0].cpu().numpy()

def update_ligand_entry(entry, model_type, label, pose, affinity):
    """
    Update ligand data dictionary with model predictions and label.
    Args:
        entry (dict): Data dictionary for a ligand
        model_type (str): Model name string
        label (float): True label value
        pose (float): Predicted pose score
        affinity (float): Predicted affinity score
    """
    entry['labels'] = label
    entry[f'pose_scores_{model_type}'] = pose
    entry[f'affinity_scores_{model_type}'] = affinity
    entry[f'sq_difference_{model_type}'] = np.square(entry['exp_value'] - affinity)

def compute_all_metrics(exp_vals, pred_vals):
    """
    Compute evaluation metrics between experimental and predicted values.
    Includes Pearson's r, Kendall's tau, Spearman's rho, RMSD, and STD of residuals.
    Args:
        exp_vals (list or np.array): Experimental/reference values
        pred_vals (list or np.array): Model predicted values
    Returns:
        dict: Metrics with keys 'Pearson', 'Kendall', 'Spearman', 'RMSD', 'STD'
    """
    if len(exp_vals) < 2:
        return {
            'Pearson': float('nan'),
            'Kendall': float('nan'),
            'Spearman': float('nan'),
            'RMSD': float('nan'),
            'STD': float('nan')
        }

    exp_arr = np.array(exp_vals)
    pred_arr = np.array(pred_vals)
    squared_diffs = (exp_arr - pred_arr) ** 2

    pearson_corr, _ = pearsonr(exp_arr, pred_arr)
    kendall_corr, _ = kendalltau(exp_arr, pred_arr)
    spearman_corr, _ = spearmanr(exp_arr, pred_arr)

    rmsd = np.sqrt(np.mean(squared_diffs))
    std = np.std(np.sqrt(squared_diffs))

    return {
        'Pearson': pearson_corr,
        'Kendall': kendall_corr,
        'Spearman': spearman_corr,
        'RMSD': rmsd,
        'STD': std
    }

def evaluate_model_on_target(group, target_folder, model_type=None, use_stored_preds=False):
    """
    Evaluate a model on a specific target/folder.
    If use_stored_preds=True, evaluates based on stored predictions instead of running models.
    Args:
        group (str): Group name (e.g. 'CDK2')
        target_folder (str): Target folder name
        model_type (str, optional): Model type ('Dense' or 'Def2018')
        use_stored_preds (bool): Whether to use stored predictions (e.g., AEV-PLIG)
    """
    target_path = os.path.join(DATA_ROOT, group, target_folder)
    receptor = os.path.join(RECEPTOR_BASE, group, target_folder, f'{target_folder}_protein.pdb')
    types_file = os.path.join(target_path, TYPES_FILENAME)
    output_dir = os.path.join(OUTPUT_ROOT, group, target_folder)
    path_key = f'{group}/{target_folder}'

    if not os.path.isfile(receptor):
        print(f"Missing receptor for {path_key}, skipping...")
        return

    if use_stored_preds:
        # Use stored predictions from FEP_data for metrics calculation
        exp_vals, pred_vals = [], []
        for ligand in FEP_data[path_key].values():
            if 'exp_value' in ligand and 'pred_value' in ligand:
                exp_vals.append(ligand['exp_value'])
                pred_vals.append(ligand['pred_value'])
        metrics = compute_all_metrics(exp_vals, pred_vals)
        if len(exp_vals) >= 2:
            target_metrics.setdefault(path_key, {})['AEV-PLIG'] = metrics
        return

    # Prepare molgrid input files and provider
    os.makedirs(output_dir, exist_ok=True)
    with open(types_file, 'w') as tf:
        references = process_molecules(
            sdf_path=os.path.join(target_path, f'{target_folder}_ligands.sdf'),
            label=1,
            prefix='active',
            output_dir=output_dir,
            receptor_path=receptor,
            types_file_handle=tf
        )

    provider = molgrid.ExampleProvider(data_root='.', balanced=False, shuffle=False, cache_structs=True)
    provider.populate(types_file)

    # Prepare grid and load model
    grid_maker, dims, tensor = prepare_grid(provider)
    model = load_model(model_type, dims)

    exp_vals, pred_vals = [], []
    # Iterate through batches and run model predictions
    for _ in range(math.ceil(provider.size() / BATCH_SIZE)):
        batch = provider.next_batch(BATCH_SIZE)
        if not batch:
            break
        labels, codes, poses, affinities = run_model_on_batch(batch, grid_maker, tensor, model)
        for i, code in enumerate(codes):
            ligand_name = references.get(code)
            if ligand_name not in FEP_data[path_key]:
                continue
            entry = FEP_data[path_key][ligand_name]
            update_ligand_entry(entry, model_type, labels[i], poses[i], affinities[i])
            exp_vals.append(entry['exp_value'])
            pred_vals.append(affinities[i])

    # Compute and store metrics for this target/model
    metrics = compute_all_metrics(exp_vals, pred_vals)
    if exp_vals:
        target_metrics.setdefault(path_key, {})[model_type] = metrics

def run_pipeline():
    """
    Run the evaluation pipeline over all groups and targets in DATA_ROOT.
    Evaluates both models ('Def2018' and 'Dense') and stored predictions ('AEV-PLIG').
    Saves updated FEP data CSV and prints summaries.
    """
    for group in sorted(os.listdir(DATA_ROOT)):
        group_path = os.path.join(DATA_ROOT, group)
        if not os.path.isdir(group_path):
            continue

        for folder in sorted(os.listdir(group_path)):
            path_key = f'{group}/{folder}'
            if path_key not in FEP_data:
                continue
            print(f"Processing target: {path_key}")

            # Evaluate Def2018 model
            evaluate_model_on_target(group, folder, model_type='Def2018', use_stored_preds=False)
            # Evaluate Dense model
            evaluate_model_on_target(group, folder, model_type='Dense', use_stored_preds=False)
            # Evaluate stored predictions (AEV-PLIG)
            evaluate_model_on_target(group, folder, use_stored_preds=True)

    save_fep_data_csv(FEP_data)
    summarize_results()

def summarize_results():
    """
    Print summary statistics (mean ± std) for all evaluated models across all targets.
    """
    all_metrics = {'Def2018': [], 'Dense': [], 'AEV-PLIG': []}
    for target, metrics in target_metrics.items():
        for model, vals in metrics.items():
            all_metrics[model].append(vals)

    print("\n==== Overall Summary ====")
    for model, entries in all_metrics.items():
        if not entries:
            continue
        print(f"\n{model} Summary:")
        for key in ['RMSD', 'Pearson', 'Kendall', 'Spearman']:
            vals = [m[key] for m in entries if not math.isnan(m[key])]
            if vals:
                print(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

def save_fep_data_csv(fep_data, filename='FEP/FEP_benchmark_updated.csv'):
    """
    Save the updated FEP data with predictions and scores to CSV.
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = [
            'target', 'ligand', 'exp_value',
            'pose_scores_Def2018', 'affinity_scores_Def2018',
            'pose_scores_Dense', 'affinity_scores_Dense'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for target, ligands in fep_data.items():
            for ligand, data in ligands.items():
                row = {
                    'target': target,
                    'ligand': ligand,
                    'exp_value': data.get('exp_value', ''),
                    'pose_scores_Def2018': data.get('pose_scores_Def2018', ''),
                    'affinity_scores_Def2018': data.get('affinity_scores_Def2018', ''),
                    'pose_scores_Dense': data.get('pose_scores_Dense', ''),
                    'affinity_scores_Dense': data.get('affinity_scores_Dense', '')
                }
                writer.writerow(row)

def generate_plots():
    """
    Generate plots for the FEP data predictions and differences.
    """
    print('\n==== Generating Plots ====')
    plot_kde_affinity_differences(FEP_data)
    plot_boxplot_with_jitter(FEP_data)

if __name__ == "__main__":
    run_pipeline()
    generate_plots()
    plot_overall_correlation(FEP_data, target_metrics)
