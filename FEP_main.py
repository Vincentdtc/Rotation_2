import os
import math
import torch
import molgrid
import numpy as np
import matplotlib.pyplot as plt
from FEP_functions import *
from gnina_dense_model import Dense
import matplotlib.cm as cm

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
        print(references)

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

# === PLOT RMSD PER TARGET WITH STD === #
targets = list(target_metrics.keys())
rmsds = [target_metrics[t]['RMSD'] for t in targets]
stds = [target_metrics[t]['STD'] for t in targets]

plt.figure(figsize=(12, 6))
plt.bar(targets, rmsds, yerr=stds, capsize=5, color='skyblue', edgecolor='black')
plt.xticks(rotation=45, ha='right')
plt.ylabel('RMSD')
plt.title('RMSD per Target (with Std Dev)')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("RMSD_per_target_with_std.png", dpi=300)
plt.show()

# === COLLECT DATA WITH TARGET LABELS === #
all_exp_affinities = []
all_pred_affinities = []
all_pose_scores = []
all_targets = []

for target in FEP_data:
    for ligand, data in FEP_data[target].items():
        if 'affinity_scores' in data and 'pose_scores' in data and 'exp_value' in data:
            all_exp_affinities.append(data['exp_value'])
            all_pred_affinities.append(data['affinity_scores'])
            all_pose_scores.append(data['pose_scores'])
            all_targets.append(target)

all_exp_affinities = np.array(all_exp_affinities)
all_pred_affinities = np.array(all_pred_affinities)
all_pose_scores = np.array(all_pose_scores)
all_targets = np.array(all_targets)

# === COLOR MAPPING BY TARGET === #
unique_targets = sorted(set(all_targets))
colors = cm.get_cmap('tab20', len(unique_targets))
target_to_color = {t: colors(i) for i, t in enumerate(unique_targets)}

# === COMBINED FIGURE WITH TWO SUBPLOTS === #
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# --- SUBPLOT 1: Predicted vs Experimental Affinity --- #
ax1 = axes[0]
for t in unique_targets:
    idxs = all_targets == t
    ax1.scatter(all_exp_affinities[idxs], all_pred_affinities[idxs],
                label=t, alpha=0.6, edgecolor='k', s=40,
                color=target_to_color[t])
m1, b1 = np.polyfit(all_exp_affinities, all_pred_affinities, 1)
ax1.plot(all_exp_affinities, m1 * all_exp_affinities + b1, color='red', linestyle='--', label='Fit')
corr1 = np.corrcoef(all_exp_affinities, all_pred_affinities)[0, 1]
ax1.set_xlabel('Experimental Affinity')
ax1.set_ylabel('Predicted Affinity')
ax1.set_title('Predicted vs Experimental Affinity')
ax1.text(0.05, 0.95, f'Pearson r = {corr1:.3f}', transform=ax1.transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
ax1.grid(True)

# --- SUBPLOT 2: Pose Score vs Predicted Affinity --- #
ax2 = axes[1]
for t in unique_targets:
    idxs = all_targets == t
    ax2.scatter(all_pose_scores[idxs], all_pred_affinities[idxs],
                label=t, alpha=0.6, edgecolor='k', s=40,
                color=target_to_color[t])
m2, b2 = np.polyfit(all_pose_scores, all_pred_affinities, 1)
ax2.plot(all_pose_scores, m2 * all_pose_scores + b2, color='red', linestyle='--', label='Fit')
corr2 = np.corrcoef(all_pose_scores, all_pred_affinities)[0, 1]
ax2.set_xlabel('Pose Score')
ax2.set_title('Pose Score vs Predicted Affinity')
ax2.text(0.05, 0.95, f'Pearson r = {corr2:.3f}', transform=ax2.transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
ax2.grid(True)

# === SHARED LEGEND BELOW === #
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, title="Target", loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.05))

# === FINAL LAYOUT AND SAVE === #
plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for legend
plt.savefig("Combined_Affinity_Correlation_Plots_Centered.png", dpi=300, bbox_inches='tight')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# === GROUPED BAR PLOT: Mean Experimental, Predicted, and Model Affinity per Target with STD === #
target_labels = []
mean_experimental_affinities = []
mean_predicted_pkds = []
mean_model_affinities = []

std_experimental_affinities = []
std_predicted_pkds = []
std_model_affinities = []

# Loop through each target and calculate mean + std
for target in sorted(FEP_data.keys()):
    exp_values = []
    pred_values = []
    model_values = []

    for ligand in sorted(FEP_data[target].keys()):
        data = FEP_data[target][ligand]
        if 'exp_value' in data and 'pred_value' in data and 'affinity_scores' in data:
            exp_values.append(data['exp_value'])
            pred_values.append(data['pred_value'])
            model_values.append(data['affinity_scores'])

    if exp_values and pred_values and model_values:
        target_labels.append(target)
        
        # Mean
        mean_experimental_affinities.append(np.mean(exp_values))
        mean_predicted_pkds.append(np.mean(pred_values))
        mean_model_affinities.append(np.mean(model_values))

        # Std
        std_experimental_affinities.append(np.std(exp_values))
        std_predicted_pkds.append(np.std(pred_values))
        std_model_affinities.append(np.std(model_values))

# Plotting
x = np.arange(len(target_labels))
width = 0.3

fig, ax = plt.subplots(figsize=(14, 6))

ax.bar(x - width, mean_experimental_affinities, width, yerr=std_experimental_affinities,
       label='Mean Experimental Affinity', color='lightblue', capsize=5)

ax.bar(x, mean_predicted_pkds, width, yerr=std_predicted_pkds,
       label='Mean Valsson et al. 2025', color='orange', capsize=5)

ax.bar(x + width, mean_model_affinities, width, yerr=std_model_affinities,
       label='Mean Predicted Affinity', color='green', capsize=5)

# Adding labels and title
ax.set_ylabel('Mean Affinity (pKd)')
ax.set_title('Mean Experimental vs Predicted vs Valsson et al. 2025 Affinity per Target')
ax.set_xticks(x)
ax.set_xticklabels(target_labels, rotation=45, ha='right', fontsize=10)
ax.legend()

# Grid and layout
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save and show
plt.savefig("Grouped_Affinity_Barplot_Mean_STD_Per_Target.png", dpi=300)
plt.show()

# # === GROUPED BAR PLOT: Experimental, Predicted, and Model Affinity per Ligand === #
# ligand_labels = []
# experimental_affinities = []
# predicted_pkds = []
# model_affinities = []

# # Loop through targets and ligands to collect the data
# for target in sorted(FEP_data.keys()):
#     for ligand in sorted(FEP_data[target].keys()):
#         data = FEP_data[target][ligand]
#         if 'exp_value' in data and 'pred_value' in data and 'affinity_scores' in data:
#             label = f"{target}/{ligand}"
#             ligand_labels.append(label)
#             experimental_affinities.append(data['exp_value'])
#             predicted_pkds.append(data['pred_value'])
#             model_affinities.append(data['affinity_scores'])

# # Plotting
# x = np.arange(len(ligand_labels))
# width = 0.3  # Width for the bars

# fig, ax = plt.subplots(figsize=(max(14, len(ligand_labels) * 0.3), 6))
# ax.bar(x - width, experimental_affinities, width, label='Experimental Affinity', color='lightblue')
# ax.bar(x, predicted_pkds, width, label='Valsson et al. 2025', color='orange')
# ax.bar(x + width, model_affinities, width, label='Predicted Affinity', color='green')

# # Adding labels and title
# ax.set_ylabel('Affinity (pKd)')
# ax.set_title('Experimental vs Predicted vs Valsson et al. 2025 Affinity per Ligand')
# ax.set_xticks(x)
# ax.set_xticklabels(ligand_labels, rotation=90, ha='center', fontsize=8)
# ax.legend()

# # Adding gridlines and adjusting layout
# ax.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()

# # Save the plot as a file and show it
# plt.savefig("Grouped_Affinity_Barplot_Per_Ligand_Experimental_Predicted_Model.png", dpi=300)
# plt.show()
