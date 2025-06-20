import os
import math
import csv
import torch
import molgrid
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, kendalltau, spearmanr

from FEP_functions import *
from FEP_plotting import *
from gnina_dense_model import Dense
from default2018_model import Net

# === CONFIGURATION === #
DATA_ROOT = 'FEP/FEP_data'
OUTPUT_ROOT = 'FEP/FEP_ligands_sdf'
WEIGHTS = {
    'Dense': './weights/dense.pt',
    'Def2018': './weights/crossdock_default2018.pt'
}
TYPES_FILENAME = 'molgrid_input.types'
RECEPTOR_BASE = DATA_ROOT
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === GLOBALS === #
FEP_data = load_fep_data('FEP/FEP_benchmark.csv')
target_metrics = {}  # Stores performance metrics for each target and model


def load_model(model_type, input_dims):
    """
    Load the model architecture and weights for the given type.

    Args:
        model_type (str): Either 'Dense' or 'Def2018'
        input_dims (tuple): Input shape from molgrid

    Returns:
        torch.nn.Module: Loaded model in evaluation mode
    """
    model = Dense(input_dims) if model_type == 'Dense' else Net(input_dims)
    model.load_state_dict(torch.load(WEIGHTS[model_type], map_location=DEVICE))
    return model.to(DEVICE).eval()


def prepare_grid(provider):
    """
    Initialize molgrid grid maker and create tensor placeholder.

    Args:
        provider (molgrid.ExampleProvider): Data provider instance

    Returns:
        Tuple of (grid_maker, grid_dims, tensor)
    """
    grid_maker = molgrid.GridMaker(resolution=0.5, dimension=23.5)
    dims = grid_maker.grid_dimensions(provider.num_types())
    tensor = torch.empty((BATCH_SIZE,) + tuple(dims), dtype=torch.float32, device=DEVICE)
    return grid_maker, dims, tensor


def run_model_on_batch(batch, grid_maker, tensor, model):
    """
    Run a single prediction batch through the model.

    Returns:
        labels, codes, pose_scores, affinity_scores
    """
    labels = torch.empty(BATCH_SIZE, dtype=torch.float32, device=DEVICE)
    codes = torch.empty(BATCH_SIZE, dtype=torch.float32, device=DEVICE)

    batch.extract_label(0, labels)
    batch.extract_label(1, codes)

    grid_maker.forward(batch, tensor, random_rotation=False, random_translation=0.0)

    with torch.no_grad():
        pose, affinity = model(tensor)

    return (
        labels.cpu().numpy(),
        codes.cpu().numpy().astype(int),
        pose.cpu().numpy(),
        affinity[:, 0].cpu().numpy()
    )


def update_ligand_entry(entry, model_type, label, pose, affinity):
    """
    Update dictionary entry with predicted scores and true label.
    """
    entry['labels'] = label
    entry[f'pose_scores_{model_type}'] = pose
    entry[f'affinity_scores_{model_type}'] = affinity
    entry[f'sq_difference_{model_type}'] = np.square(entry['exp_value'] - affinity)


def compute_all_metrics(exp_vals, pred_vals):
    """
    Compute various correlation and error metrics.

    Returns:
        dict with Pearson, Kendall, Spearman, RMSD, STD
    """
    if len(exp_vals) < 2:
        return {k: float('nan') for k in ['Pearson', 'Kendall', 'Spearman', 'RMSD', 'STD']}

    exp = np.array(exp_vals)
    pred = np.array(pred_vals)
    diff = exp - pred

    return {
        'Pearson': pearsonr(exp, pred)[0],
        'Kendall': kendalltau(exp, pred)[0],
        'Spearman': spearmanr(exp, pred)[0],
        'RMSD': np.sqrt(np.mean(diff ** 2)),
        'STD': np.std(diff)
    }


def evaluate_model_on_target(group, target_folder, model_type=None, use_stored_preds=False):
    """
    Evaluate a specific target using either a model or stored predictions.
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
        exp_vals, pred_vals = [], []
        for ligand in FEP_data[path_key].values():
            if 'exp_value' in ligand and 'pred_value' in ligand:
                exp_vals.append(ligand['exp_value'])
                pred_vals.append(ligand['pred_value'])
        metrics = compute_all_metrics(exp_vals, pred_vals)
        if exp_vals:
            target_metrics.setdefault(path_key, {})['AEV-PLIG'] = metrics
        return

    # Prepare data and model
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

    grid_maker, dims, tensor = prepare_grid(provider)
    model = load_model(model_type, dims)

    exp_vals, pred_vals = [], []

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

    metrics = compute_all_metrics(exp_vals, pred_vals)
    if exp_vals:
        target_metrics.setdefault(path_key, {})[model_type] = metrics


def run_pipeline():
    """
    Evaluate all targets for each model and stored predictions.
    Saves updated CSV and generates summary.
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

            evaluate_model_on_target(group, folder, model_type='Def2018')
            evaluate_model_on_target(group, folder, model_type='Dense')
            evaluate_model_on_target(group, folder, use_stored_preds=True)

    save_fep_data_csv(FEP_data)
    summarize_results()


def summarize_results():
    """
    Print mean ± std summary of key metrics for each model.
    """
    all_metrics = {'Def2018': [], 'Dense': [], 'AEV-PLIG': []}

    for metrics in target_metrics.values():
        for model, vals in metrics.items():
            all_metrics[model].append(vals)

    print("\n==== Overall Summary ====")
    for model, entries in all_metrics.items():
        if not entries:
            continue
        print(f"\n{model} Summary:")
        for key in ['RMSD', 'Pearson', 'Kendall', 'Spearman']:
            values = [m[key] for m in entries if not math.isnan(m[key])]
            if values:
                print(f"  {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")


def save_fep_data_csv(fep_data, filename='FEP/FEP_benchmark_updated.csv'):
    """
    Save FEP data with predictions into a CSV file.

    Args:
        fep_data (dict): Nested dictionary with ligand prediction data
        filename (str): Output path
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
                writer.writerow({
                    'target': target,
                    'ligand': ligand,
                    'exp_value': data.get('exp_value', ''),
                    'pose_scores_Def2018': data.get('pose_scores_Def2018', ''),
                    'affinity_scores_Def2018': data.get('affinity_scores_Def2018', ''),
                    'pose_scores_Dense': data.get('pose_scores_Dense', ''),
                    'affinity_scores_Dense': data.get('affinity_scores_Dense', '')
                })


def generate_plots():
    """
    Generate and display plots for prediction comparisons.
    """
    print('\n==== Generating Plots ====')
    plot_kde_affinity_differences(FEP_data)
    plot_boxplot_with_jitter(FEP_data)
    plot_overall_correlation(FEP_data, target_metrics)


# === ENTRY POINT === #
if __name__ == "__main__":
    run_pipeline()
    generate_plots()
