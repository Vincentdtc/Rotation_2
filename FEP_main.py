import os
import math
import torch
import molgrid
import numpy as np
import matplotlib.pyplot as plt
from FEP_functions import *
from gnina_dense_model import Dense
from plotting import *

# === CONFIGURATION SECTION === #
DATA_ROOT = 'FEP_data'
OUTPUT_ROOT = 'FEP_ligands_sdf'
WEIGHTS_PATH = './weights/dense.pt'
TYPES_FILENAME = 'molgrid_input.types'
RECEPTOR_BASE = DATA_ROOT
BATCH_SIZE = 1
top_n = 3
method = 'max_aff'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === GLOBAL VARIABLES === #
model = None
target_metrics = {}
per_target_data = {}

FEP_data = load_fep_data('FEP_benchmark.csv')

# === UTILITY FUNCTIONS === #
def load_model(input_dims):
    m = Dense(input_dims).to(DEVICE)
    m.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    m.eval()
    return m

def prepare_gridmaker_and_tensor(provider):
    grid_maker = molgrid.GridMaker(resolution=0.5, dimension=23.5)
    dims = grid_maker.grid_dimensions(provider.num_types())
    tensor_shape = (BATCH_SIZE,) + tuple(dims)
    tensor = torch.empty(tensor_shape, dtype=torch.float32, device=DEVICE)
    return grid_maker, dims, tensor

def process_target(target_folder):
    target_path = os.path.join(DATA_ROOT, target_folder)
    receptor_path = os.path.join(RECEPTOR_BASE, target_folder, f'{target_folder}_protein.pdb')
    types_file = os.path.join(target_path, TYPES_FILENAME)
    output_dir = os.path.join(OUTPUT_ROOT, target_folder)

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

    provider = molgrid.ExampleProvider(data_root='.', balanced=False, shuffle=False, cache_structs=True)
    provider.populate(types_file)

    grid_maker, dims, tensor = prepare_gridmaker_and_tensor(provider)

    global model
    if model is None:
        model = load_model(dims)

    float_labels = torch.empty(BATCH_SIZE, dtype=torch.float32, device=DEVICE)
    float_codes = torch.empty(BATCH_SIZE, dtype=torch.float32, device=DEVICE)

    num_batches = math.ceil(provider.size() / BATCH_SIZE)
    squared_diffs = []

    for _ in range(num_batches):
        batch = provider.next_batch(BATCH_SIZE)
        if batch is None:
            break

        batch.extract_label(0, float_labels)
        batch.extract_label(1, float_codes)

        grid_maker.forward(batch, tensor, random_rotation=False, random_translation=0.0)
        with torch.no_grad():
            pose, affinity = model(tensor)

        labels_np = float_labels.cpu().numpy()
        codes_np = float_codes.cpu().numpy().astype(np.int64)
        poses_np = pose.cpu().numpy()
        affinities_np = affinity[:, 0].cpu().numpy()

        for i, code in enumerate(codes_np):
            name = references[code]
            FEP_data[target_folder][name]['labels'] = labels_np[i]
            FEP_data[target_folder][name]['pose_scores'] = poses_np[i]
            FEP_data[target_folder][name]['affinity_scores'] = affinities_np[i]
            sq_diff = np.power(FEP_data[target_folder][name]['exp_value'] - affinities_np[i], 2)
            FEP_data[target_folder][name]['sq_difference'] = sq_diff
            squared_diffs.append(sq_diff)

    squared_diffs = np.array(squared_diffs)
    RMSD = np.sqrt(np.mean(squared_diffs))
    STD = np.std(np.sqrt(squared_diffs))

    target_metrics[target_folder] = {
        'num_ligands': len(FEP_data[target_folder].keys()),
        'RMSD': RMSD,
        'STD': STD
    }

# === MAIN EXECUTION LOOP === #
for folder in sorted(os.listdir(DATA_ROOT)):
    if os.path.isdir(os.path.join(DATA_ROOT, folder)):
        print(f"\nProcessing target: {folder}")
        process_target(folder)

print("\n==== Summary of Metrics by Target ====")
overall_RMSD = []
for target, metrics in target_metrics.items():
    print(f"\nTarget: {target}")
    for key, value in metrics.items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
                if key == 'RMSD':
                    overall_RMSD.append(value)
            else:
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: N/A")
print(f"\nOverall RMSD: {np.mean(overall_RMSD):.4f} ± {np.std(overall_RMSD):.4f}")
print("==== End of Summary ====")

# === PLOTTING SECTION === #
plot_rmsd_per_target(target_metrics)
save_per_target_correlation(FEP_data)
plot_grouped_affinity(FEP_data)

handles, labels = plot_combined_correlation(FEP_data)
save_legend(handles, labels)
plt.show()
