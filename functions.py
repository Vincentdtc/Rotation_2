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
import pandas as pd

def ensure_output_dir(base_dir='results_DUD_E'):
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

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
            print(f"{chembl_id}: only {chembl_counts[chembl_id]} conformers found")

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

def plot_ef_nef_grouped_bar(target_metrics, output_dir='results_DUD_E'):
    """
    Plot grouped bar charts of EF and NEF values at 1%, 5%, and 10% across all targets.
    One figure for EF and one for NEF.

    Saves:
    - grouped_ef_plot.png in the output_dir
    - grouped_nef_plot.png in the output_dir
    """
    # Ensure the output directory exists
    output_dir = ensure_output_dir(output_dir)
    sns.set_theme(style="whitegrid")

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
    fig_ef.savefig(os.path.join(output_dir, "grouped_ef_plot.png"), dpi=300)

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
    fig_nef.savefig(os.path.join(output_dir, "grouped_nef_plot.png"), dpi=300)

    print(f"\nSaved grouped EF plot to: {os.path.join(output_dir, 'grouped_ef_plot.png')}")
    print(f"Saved grouped NEF plot to: {os.path.join(output_dir, 'grouped_nef_plot.png')}")

def plot_results(per_target_data, 
                 output_dir='results_DUD_E',
                 roc_outfile="roc_curves_pose.png", 
                 dist_outfile="affinity_distributions.png", 
                 affinity_roc_outfile="roc_curves_affinity.png"):
    """
    Plot ROC curves (pose and affinity), and affinity distribution histograms for each target.
    
    Saves:
    - roc_curves_pose.png in the output_dir
    - affinity_distributions.png in the output_dir
    - roc_curves_affinity.png in the output_dir
    """

    # Ensure the output directory exists
    output_dir = ensure_output_dir(output_dir)

    num_targets = len(per_target_data)
    cols = 3
    rows = (num_targets + cols - 1) // cols

    fig_roc_pose, axs_roc_pose = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    fig_roc_affinity, axs_roc_affinity = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    fig_dist, axs_dist = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))

    axs_roc_pose = axs_roc_pose.flatten()
    axs_roc_affinity = axs_roc_affinity.flatten()
    axs_dist = axs_dist.flatten()

    for idx, (target, data) in enumerate(per_target_data.items()):
        labels = data['labels']
        poses = data['pose_scores']
        affinities = data['affinity_scores']

        # --- ROC Curve using pose_scores --- #
        ax_pose = axs_roc_pose[idx]
        if len(set(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, poses)
            auc = roc_auc_score(labels, poses)
            ax_pose.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color='blue')
        else:
            ax_pose.text(0.5, 0.5, "Insufficient label variation", ha='center', va='center')
        ax_pose.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax_pose.set_title(f"{target} (Pose)")
        ax_pose.set_xlabel('FPR')
        ax_pose.set_ylabel('TPR')
        ax_pose.legend()
        ax_pose.grid(True)

        # --- ROC Curve using affinity_scores --- #
        ax_aff = axs_roc_affinity[idx]
        if len(set(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, affinities)
            auc = roc_auc_score(labels, affinities)
            ax_aff.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color='purple')
        else:
            ax_aff.text(0.5, 0.5, "Insufficient label variation", ha='center', va='center')
        ax_aff.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax_aff.set_title(f"{target} (Affinity)")
        ax_aff.set_xlabel('FPR')
        ax_aff.set_ylabel('TPR')
        ax_aff.legend()
        ax_aff.grid(True)

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
    for ax in axs_roc_pose[num_targets:]:
        fig_roc_pose.delaxes(ax)
    for ax in axs_roc_affinity[num_targets:]:
        fig_roc_affinity.delaxes(ax)
    for ax in axs_dist[num_targets:]:
        fig_dist.delaxes(ax)

    fig_roc_pose.tight_layout()
    fig_roc_affinity.tight_layout()
    fig_dist.tight_layout()

    # Save the figures in the specified output directory
    fig_roc_pose.savefig(os.path.join(output_dir, roc_outfile), dpi=300)
    fig_roc_affinity.savefig(os.path.join(output_dir, affinity_roc_outfile), dpi=300)
    fig_dist.savefig(os.path.join(output_dir, dist_outfile), dpi=300)

    print(f"\nSaved pose-based ROC curves to: {os.path.join(output_dir, roc_outfile)}")
    print(f"Saved affinity-based ROC curves to: {os.path.join(output_dir, affinity_roc_outfile)}")
    print(f"Saved affinity distributions to: {os.path.join(output_dir, dist_outfile)}")

def compute_roc_enrichment_factors(y_true, y_score, fpr_levels=[0.01, 0.02, 0.05]):
    """
    Compute ROC enrichment factors at specified false-positive rates.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_score : array-like of shape (n_samples,)
        Target scores, probabilities, or decision function.
    fpr_levels : list of floats, optional (default=[0.01, 0.02, 0.05])
        FPR cutoffs at which to compute enrichment (e.g. 0.01 == 1%).

    Returns
    -------
    enrichment : dict
        Mapping each FPR cutoff to its enrichment factor, defined as
            EF(x) = TPR(x) / x
        where TPR(x) is the true-positive rate at FPR = x.
    """
    # Compute full ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)

    enrichment = {}
    for x in fpr_levels:
        if not (0 < x < 1):
            raise ValueError(f"FPR level must be between 0 and 1, got {x}")
        # Interpolate TPR at the desired FPR
        tpr_at_x = np.interp(x, fpr, tpr)
        enrichment[x] = tpr_at_x / x
    return enrichment

def plot_roc_ef_grouped_bar(target_metrics, fpr_levels=[0.005, 0.01, 0.02, 0.05], save_path='roc_ef_grouped_bar.png', output_dir='results_DUD_E'):
    """
    Plot ROC Enrichment Factors (EFs) grouped by target in a single figure and save it.

    Parameters:
    - target_metrics (dict): Dictionary of target metrics.
    - fpr_levels (list): List of FPR levels to plot (e.g., [0.005, 0.01, 0.02, 0.05]).
    - save_path (str): Path to save the generated figure.
    - output_dir (str): Directory where the figure should be saved.
    """

    # Ensure the output directory exists
    output_dir = ensure_output_dir(output_dir)

    # Create dataframe for plotting
    data = []
    for target, metrics in target_metrics.items():
        for fpr in fpr_levels:
            ef_key = f'ROC EF {fpr*100:.1f}%'
            ef_value = metrics.get(ef_key)
            if ef_value is not None:
                data.append({'Target': target, 'FPR Level': f'{fpr*100:.1f}%', 'EF': ef_value})

    df = pd.DataFrame(data)

    # Plot using seaborn
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df, x='Target', y='EF', hue='FPR Level', palette='viridis')

    plt.xticks(rotation=45, ha='right')
    plt.title('ROC Enrichment Factors Grouped by Target')
    plt.xlabel('Target')
    plt.ylabel('Enrichment Factor (EF)')
    plt.legend(title='FPR Level')
    plt.tight_layout()

    # Ensure save path is inside the specified output directory
    save_path = os.path.join(output_dir, save_path)

    # Save figure
    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to {save_path}")

    # Show figure
    # plt.show()
