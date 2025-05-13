# Imports
import os
import numpy as np
from rdkit import Chem
from matplotlib import pyplot as plt
import math

from rdkit import Chem
import os

def process_molecules(sdf_path, label, prefix, output_dir, receptor_path, types_file_handle):
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    index_to_title = {}  # Dictionary to store index-to-title mapping

    for i, mol in enumerate(supplier):
        if mol is None:
            continue

        # Extract molecule identifier from the title
        title = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i}"
        chembl_id = title.strip().split()[0]

        # Set the filename and path for the ligand
        ligand_filename = f'{prefix}_{chembl_id}_{i}.sdf'
        ligand_path = os.path.join(output_dir, ligand_filename)

        try:
            Chem.MolToMolFile(mol, ligand_path)
        except Exception as e:
            print(f"Error writing molecule {i} ({chembl_id}): {e}")
            continue

        # Write to the types file
        types_file_handle.write(f'{label} {i} {receptor_path} {ligand_path}\n')

        # Store mapping
        index_to_title[i] = title

    print("Molecule processing complete.")
    return index_to_title


def deltaG_to_pKd(deltaG_kcal, temperature=297):
    """
    Convert Gibbs free energy (ΔG, in kcal/mol) to pKd (–log10 of the dissociation constant).
    
    Parameters:
        deltaG_kcal (float): Gibbs free energy change (negative for favorable binding), in kcal/mol.
        temperature (float): Temperature in Kelvin (default is 298.15 K).

    Returns:
        float: pKd (dimensionless).
    """
    R = 1.98720425864083e-3  # kcal/mol·K taken from McNutt and Koes 2022 (as well as T value).
    Kd = math.exp(deltaG_kcal / (R * temperature))
    pKd = -math.log10(Kd)
    return pKd

import pandas as pd
from FEP_functions import deltaG_to_pKd

def load_fep_data(file_path):
    """
    Loads FEP benchmark data from a CSV file and returns a nested dictionary
    mapping protein names to ligand names and their corresponding pKd values.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        dict: A nested dictionary of the form {protein: {ligand: pKd}}.
    """
    df = pd.read_csv(file_path)
    df['protein'] = df['group_id'].apply(lambda x: x.split('/')[-1])

    result = {}
    for _, row in df.iterrows():
        protein = row['protein']
        ligand = row['Ligand name']
        exp_dG = row['Exp. dG (kcal/mol)']
        
        if protein not in result:
            result[protein] = {}
        
        result[protein][ligand] = {'exp_value': deltaG_to_pKd(exp_dG)}

    return result

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
