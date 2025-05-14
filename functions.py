# Imports
import shutil
import os
import re
import numpy as np
from rdkit import Chem
import gzip
from collections import defaultdict
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

def extract_sdf_gz_files(directory):
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        # Check if the specific file exists
        for file in files:
            if file.endswith("_docked_vina.sdf.gz"):
                # Construct the full path to the file
                file_path = os.path.join(root, file)
                
                # Define the output file path (remove .gz from the name)
                output_file_path = os.path.join(root, file.replace('.gz', ''))
                
                # Extract the file
                with gzip.open(file_path, 'rb') as f_in:
                    with open(output_file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                print(f"Extracted: {file_path} -> {output_file_path}")

def process_molecules(sdf_path, number, label, prefix, output_dir, receptor_path, types_file_handle, batch_num=0):
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)

    chembl_counts = defaultdict(int)     # Tracks count per ChEMBL ID
    seen_chembl_ids = set()              # To remember which IDs were seen

    for i, mol in enumerate(supplier):
        if mol is None:
            continue

        # Extract ChEMBL ID from molecule title
        title = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i}"
        chembl_id = title.strip().split()[0]

        seen_chembl_ids.add(chembl_id)

        # Only keep up to `number` molecules per ChEMBL ID
        if chembl_counts[chembl_id] >= number:
            continue

        # Set the filename and path for the ligand based on prefix
        if 'decoy' in prefix:
            ligand_filename = f'{prefix}_batch{batch_num}_{chembl_id}_{chembl_counts[chembl_id]}.sdf'
        elif 'active' in prefix:
            ligand_filename = f'{prefix}_{chembl_id}_{chembl_counts[chembl_id]}.sdf'
        else:
            raise ValueError("Prefix must contain 'decoy' or 'active' to specify molecule type.")

        ligand_path = os.path.join(output_dir, ligand_filename)

        try:
            Chem.MolToMolFile(mol, ligand_path)
        except Exception as e:
            print(f"Error writing molecule {i} ({chembl_id}) in batch {batch_num}: {e}")
            continue

        # Extract a unique numeric code from the ChEMBL ID (if possible)
        unique_code = int(re.findall(r'\d+', chembl_id)[0])

        # Write to the types file
        types_file_handle.write(f'{label} {unique_code} {receptor_path} {ligand_path}\n')

        chembl_counts[chembl_id] += 1

    # Report ChEMBL IDs with fewer than `number` entries
    for chembl_id in seen_chembl_ids:
        if chembl_counts[chembl_id] < number:
            print(f"{chembl_id}: only {chembl_counts[chembl_id]} molecules found")

def compute_enrichment_factors(all_affinities, all_labels, level):
    """
    Compute the Enrichment Factor (EF) and Normalized Enrichment Factor (NEF) at a given percentage level.

    Parameters:
    ----------
    all_affinities : array-like
        Predicted affinities or scores for all compounds (higher = better).
    all_labels : array-like
        Binary activity labels (1 = active, 0 = inactive) for each compound.
    level : float
        Percentage level (e.g., 1 for top 1%) at which to compute EF and NEF.

    Returns:
    -------
    nef : float
        Normalized Enrichment Factor at the specified level.
    ef : float
        Raw Enrichment Factor at the specified level.
    """
    all_affinities = np.asarray(all_affinities)
    all_labels = np.asarray(all_labels)

    total = len(all_labels)
    num_actives = np.sum(all_labels)

    if total == 0 or num_actives == 0:
        # No compounds or no actives — EF/NEF are undefined
        return 0.0, 0.0

    # Determine number of top compounds to evaluate (top 'level'%)
    top_n = total * level // 100
    if top_n < 1:
        return 0.0, 0.0

    # Get indices of top 'top_n' compounds by descending affinity
    top_indices = np.argpartition(-all_affinities, top_n - 1)[:top_n]

    # Count actives in top subset
    top_actives = np.sum(all_labels[top_indices])

    # Calculate Enrichment Factor (EF)
    expected_actives = num_actives * (level / 100)
    ef = top_actives / expected_actives if expected_actives > 0 else 0.0

    # Calculate maximum possible EF (ideal scenario)
    max_top_actives = min(num_actives, top_n)
    max_ef = max_top_actives / expected_actives if expected_actives > 0 else 0.0

    # Normalized Enrichment Factor
    nef = ef / max_ef if max_ef > 0 else 0.0

    return nef, ef

def plot_ef_nef_grouped_bar(target_metrics):
    """
    Plot grouped bar charts of EF and NEF values at 1%, 5%, and 10% across all targets.
    One figure for EF and one for NEF.

    Saves:
    - grouped_ef_plot.png
    - grouped_nef_plot.png
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")

    # Extract target names
    targets = list(target_metrics.keys())

    # Extract EF and NEF values for each level
    ef_levels = ['EF 1%', 'EF 5%', 'EF 10%']
    nef_levels = ['NEF 1%', 'NEF 5%', 'NEF 10%']

    ef_data = {level: [target_metrics[t][level] for t in targets] for level in ef_levels}
    nef_data = {level: [target_metrics[t][level] for t in targets] for level in nef_levels}

    x = np.arange(len(targets))  # positions for the bars
    width = 0.25  # width of each bar

    # === EF Plot === #
    fig_ef, ax_ef = plt.subplots(figsize=(max(10, len(targets) * 0.6), 6))
    for i, level in enumerate(ef_levels):
        ax_ef.bar(x + i * width, ef_data[level], width, label=level.upper())

    ax_ef.set_ylabel('Enrichment Factor (EF)')
    ax_ef.set_title('EF at 1%, 5%, and 10% by Target')
    ax_ef.set_xticks(x + width)
    ax_ef.set_xticklabels(targets, rotation=45, ha='right')
    ax_ef.legend()
    ax_ef.grid(True, axis='y', linestyle='--', alpha=0.6)
    fig_ef.tight_layout()
    fig_ef.savefig("grouped_ef_plot.png", dpi=300)

    # === NEF Plot === #
    fig_nef, ax_nef = plt.subplots(figsize=(max(10, len(targets) * 0.6), 6))
    for i, level in enumerate(nef_levels):
        ax_nef.bar(x + i * width, nef_data[level], width, label=level.upper())

    ax_nef.set_ylabel('Normalized Enrichment Factor (NEF)')
    ax_nef.set_title('NEF at 1%, 5%, and 10% by Target')
    ax_nef.set_xticks(x + width)
    ax_nef.set_xticklabels(targets, rotation=45, ha='right')
    ax_nef.legend()
    ax_nef.grid(True, axis='y', linestyle='--', alpha=0.6)
    fig_nef.tight_layout()
    fig_nef.savefig("grouped_nef_plot.png", dpi=300)

    print("\nSaved grouped EF plot to: grouped_ef_plot.png")
    print("Saved grouped NEF plot to: grouped_nef_plot.png")

def plot_results(per_target_data, roc_outfile="roc_curves.png", dist_outfile="affinity_distributions.png"):
    """
    Plot ROC curves and affinity distribution histograms for each target.

    Parameters:
    - per_target_data (dict): Dictionary where keys are target names and values are dictionaries with keys:
        - 'labels': list of true binary labels
        - 'pose_scores': list of predicted pose scores
        - 'affinity_scores': list of predicted affinity scores
    - roc_outfile (str): Filename to save the ROC curves plot
    - dist_outfile (str): Filename to save the affinity distributions plot
    """

    num_targets = len(per_target_data)
    cols = 3
    rows = (num_targets + cols - 1) // cols

    fig_roc, axs_roc = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    fig_dist, axs_dist = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))

    axs_roc = axs_roc.flatten()
    axs_dist = axs_dist.flatten()

    for idx, (target, data) in enumerate(per_target_data.items()):
        labels = data['labels']
        poses = data['pose_scores']
        affinities = data['affinity_scores']

        # --- ROC Curve --- #
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

        # --- Affinity Distribution --- #
        ax_dist = axs_dist[idx]
        actives = [a for a, l in zip(affinities, labels) if l == 1]
        decoys = [a for a, l in zip(affinities, labels) if l == 0]

        sns.histplot(actives, kde=True, color='green', label='Actives',
                     stat="density", ax=ax_dist, bins=20)
        sns.histplot(decoys, kde=True, color='red', label='Decoys',
                     stat="density", ax=ax_dist, bins=20)

        ax_dist.set_title(target)
        ax_dist.set_xlabel('Predicted Affinity')
        ax_dist.set_ylabel('Density')
        ax_dist.legend()
        ax_dist.grid(True)

    # Remove unused subplots
    for ax in axs_roc[num_targets:]:
        fig_roc.delaxes(ax)
    for ax in axs_dist[num_targets:]:
        fig_dist.delaxes(ax)

    fig_roc.tight_layout()
    fig_dist.tight_layout()

    fig_roc.savefig(roc_outfile, dpi=300)
    fig_dist.savefig(dist_outfile, dpi=300)

    print(f"\nSaved ROC curves to: {roc_outfile}")
    print(f"Saved affinity distributions to: {dist_outfile}")
