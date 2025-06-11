import os
import math
import torch
import molgrid
import numpy as np
from FEP_functions import *
from gnina_dense_model1 import Dense
from default2018_model import *
from FEP_plotting import *
import csv

# === CONFIGURATION === #
DATA_ROOT = 'FEP/FEP_data'
OUTPUT_ROOT = 'FEP/FEP_ligands_sdf'
WEIGHTS_PATH = './weights/dense.pt'
WEIGHTS_PATH2 = './weights/crossdock_default2018.pt'
TYPES_FILENAME = 'molgrid_input.types'
RECEPTOR_BASE = DATA_ROOT
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === GLOBAL STATE === #
FEP_data = load_fep_data('FEP/FEP_benchmark.csv')
model_cache = {}
target_metrics = {}



def save_fep_data_csv(fep_data, filename='FEP/FEP_benchmark_updated.csv'):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['target', 'ligand', 'exp_value',
                      'pose_scores_Def2018', 'affinity_scores_Def2018',
                      'pose_scores_Dense', 'affinity_scores_Dense']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for target, ligands in fep_data.items():
            for ligand_name, data in ligands.items():
                row = {
                    'target': target,
                    'ligand': ligand_name,
                    'exp_value': data.get('exp_value', ''),
                    'pose_scores_Def2018': data.get('pose_scores_Def2018', ''),
                    'affinity_scores_Def2018': data.get('affinity_scores_Def2018', ''),
                    'pose_scores_Dense': data.get('pose_scores_Dense', ''),
                    'affinity_scores_Dense': data.get('affinity_scores_Dense', '')
                }
                writer.writerow(row)


# === MODEL + GRID SETUP === #
def load_model_Dense(input_dims):
    model = Dense(input_dims).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    return model

def load_model_Def2018(input_dims):
    model = Net(input_dims).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH2, map_location=DEVICE))
    model.eval()
    return model


def prepare_grid_setup(provider):
    grid_maker = molgrid.GridMaker(resolution=0.5, dimension=23.5)
    dims = grid_maker.grid_dimensions(provider.num_types())
    tensor = torch.empty((BATCH_SIZE,) + tuple(dims), dtype=torch.float32, device=DEVICE)
    return grid_maker, dims, tensor


def get_provider(types_file):
    provider = molgrid.ExampleProvider(data_root='.', balanced=False, shuffle=False, cache_structs=True)
    provider.populate(types_file)
    return provider


# === PROCESSING === #
def process_batch(batch, grid_maker, tensor, model):
    float_labels = torch.empty(BATCH_SIZE, dtype=torch.float32, device=DEVICE)
    float_codes = torch.empty(BATCH_SIZE, dtype=torch.float32, device=DEVICE)

    batch.extract_label(0, float_labels)
    batch.extract_label(1, float_codes)
    grid_maker.forward(batch, tensor, random_rotation=False, random_translation=0.0)

    with torch.no_grad():
        pose, affinity = model(tensor)

    return (
        float_labels.cpu().numpy(),
        float_codes.cpu().numpy().astype(np.int64),
        pose.cpu().numpy(),
        affinity[:, 0].cpu().numpy()
    )


def update_fep_data(path, references, codes, labels, poses, affinities, model_type):
    squared_diffs = []
    for i, code in enumerate(codes):
        name = references.get(code)
        try:
            entry = FEP_data[path][name]
            if model_type == 'Def2018':
                entry.update({
                    'labels': labels[i],
                    'pose_scores_Def2018': poses[i],
                    'affinity_scores_Def2018': affinities[i],
                    'sq_difference_Def2018': np.power(entry['exp_value'] - affinities[i], 2)
                    })
                squared_diffs.append(entry['sq_difference_Def2018'])
            elif model_type == 'Dense':
                entry.update({
                    'labels': labels[i],
                    'pose_scores_Dense': poses[i],
                    'affinity_scores_Dense': affinities[i],
                    'sq_difference_Dense': np.power(entry['exp_value'] - affinities[i], 2)
                    })
                squared_diffs.append(entry['sq_difference_Dense'])
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except KeyError:
            print(f"Warning: {name} not found in FEP_data for {path}, skipping...")
    return squared_diffs


def process_target(group, target_folder, model_type):
    target_path = os.path.join(DATA_ROOT, group, target_folder)
    receptor_path = os.path.join(RECEPTOR_BASE, group, target_folder, f'{target_folder}_protein.pdb')
    types_file = os.path.join(target_path, TYPES_FILENAME)
    output_dir = os.path.join(OUTPUT_ROOT, group, target_folder)
    path_key = f'{group}/{target_folder}'

    if not os.path.isfile(receptor_path):
        print(f"Missing receptor for {target_folder}, skipping...")
        return

    os.makedirs(output_dir, exist_ok=True)
    with open(types_file, 'w') as tf:
        references = process_molecules(
            sdf_path=os.path.join(target_path, f'{target_folder}_ligands.sdf'),
            label=1,
            prefix='active',
            output_dir=output_dir,
            receptor_path=receptor_path,
            types_file_handle=tf
        )

    provider = get_provider(types_file)
    grid_maker, dims, tensor = prepare_grid_setup(provider)

    if model_type == 'Def2018':
        model = load_model_Def2018(dims)
    elif model_type == 'Dense':
        model = load_model_Dense(dims)

    squared_diffs = []
    for _ in range(math.ceil(provider.size() / BATCH_SIZE)):
        batch = provider.next_batch(BATCH_SIZE)
        if batch is None:
            break
        labels, codes, poses, affinities = process_batch(batch, grid_maker, tensor, model)
        squared_diffs += update_fep_data(path_key, references, codes, labels, poses, affinities, model_type)

    path_key = f'{group}/{target_folder}'

    if path_key not in target_metrics:
        target_metrics[path_key] = {}

    if squared_diffs:
        rmsd = np.sqrt(np.mean(squared_diffs))
        std = np.std(np.sqrt(squared_diffs))
        target_metrics[path_key][model_type] = {
            'num_ligands': len(FEP_data[path_key]),
            'RMSD': rmsd,
            'STD': std
        }


# === MAIN LOOP === #
def run_pipeline():
    for group in sorted(os.listdir(DATA_ROOT)):
        group_path = os.path.join(DATA_ROOT, group)
        for folder in sorted(os.listdir(group_path)):
            path_key = f'{group}/{folder}'
            if path_key in FEP_data:
                print(f"\nProcessing target: {path_key}")
                process_target(group, folder, model_type='Def2018')
                process_target(group, folder, model_type='Dense')
            else:
                print(f"Skipping {path_key}, not found in FEP_data.")
    # Save updated data with predictions from both models
    save_fep_data_csv(FEP_data)


    print("\n==== Summary of Metrics by Target ====")

    # Containers for all RMSD and STD values by method
    rmsd_def2018 = []
    std_def2018 = []
    rmsd_dense = []
    std_dense = []

    for target, metrics in target_metrics.items():
        print(f"\nTarget: {target}")
        for method in ['Def2018', 'Dense']:
            if method in metrics:
                method_metrics = metrics[method]
                print(f"  {method}:")
                for key, value in method_metrics.items():
                    print(f"    {key}: {value:.4f}")
                if 'RMSD' in method_metrics:
                    if method == 'Def2018':
                        rmsd_def2018.append(method_metrics['RMSD'])
                    else:
                        rmsd_dense.append(method_metrics['RMSD'])
                if 'STD' in method_metrics:
                    if method == 'Def2018':
                        std_def2018.append(method_metrics['STD'])
                    else:
                        std_dense.append(method_metrics['STD'])

    # Print overall summary
    print("\n==== Overall Summary ====")
    if rmsd_def2018:
        print(f"Def2018 RMSD: {np.mean(rmsd_def2018):.4f} ± {np.std(rmsd_def2018):.4f}")
    if std_def2018:
        print(f"Def2018 STD:  {np.mean(std_def2018):.4f} ± {np.std(std_def2018):.4f}")
    if rmsd_dense:
        print(f"Dense RMSD:   {np.mean(rmsd_dense):.4f} ± {np.std(rmsd_dense):.4f}")
    if std_dense:
        print(f"Dense STD:    {np.mean(std_dense):.4f} ± {np.std(std_dense):.4f}")
    print("==== End of Summary ====")


# === PLOTTING === #
def generate_plots():
    print('\n==== Generating Plots ====')
    plot_kde_affinity_differences(FEP_data)
    plot_boxplot_with_jitter(FEP_data)


if __name__ == "__main__":
    run_pipeline()
    generate_plots()
