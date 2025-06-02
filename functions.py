# Imports
import shutil
import os
import re
import numpy as np
from rdkit import Chem, RDLogger
import gzip
from collections import defaultdict
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

def ensure_output_dir(base_dir='results_DUD_E'):
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def extract_sdf_gz_files(directory, ending):
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        # Check if the specific file exists
        for file in files:
            if file.endswith(ending):
                # Construct the full path to the file
                file_path = os.path.join(root, file)
                
                # Define the output file path (remove .gz from the name)
                output_file_path = os.path.join(root, file.replace('.gz', ''))
                
                # Extract the file
                with gzip.open(file_path, 'rb') as f_in:
                    with open(output_file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

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

    # # Report ChEMBL IDs with fewer than `number` entries
    # for chembl_id in seen_chembl_ids:
    #     if chembl_counts[chembl_id] < number:
    #         print(f"{chembl_id}: only {chembl_counts[chembl_id]} conformers found")

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

def compute_ef(y_true, y_score, fpr_levels=[0.01, 0.02, 0.05]):
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

def compute_nef(all_labels, all_affinities, fpr_levels):
    all_affinities = np.asarray(all_affinities)
    all_labels = np.asarray(all_labels)

    total = len(all_labels)
    num_actives = np.sum(all_labels)

    nef_dict = {}

    # Edge case: empty input or no actives
    if total == 0 or num_actives == 0:
        return {level: 0.0 for level in fpr_levels}

    for level in fpr_levels:
        # Ensure level is a float between 0 and 1
        if not (0 < level <= 1):
            nef_dict[level] = 0.0
            continue

        top_n = int(total * level)

        if top_n < 1:
            nef_dict[level] = 0.0
            continue

        # Clamp top_n to avoid out-of-bounds or invalid indexing
        top_n = min(top_n, total)

        # Get indices of top_n highest affinity scores
        top_indices = np.argpartition(-all_affinities, top_n - 1)[:top_n]
        top_actives = np.sum(all_labels[top_indices])
        expected_actives = num_actives * level

        ef = top_actives / expected_actives if expected_actives > 0 else 0.0
        max_top_actives = min(num_actives, top_n)
        max_ef = max_top_actives / expected_actives if expected_actives > 0 else 0.0

        nef = ef / max_ef if max_ef > 0 else 0.0
        nef_dict[level] = nef

    return nef_dict

def bootstrap_roc_auc(labels, scores, n_bootstraps=1000, seed=42):
    """Bootstrap ROC AUC scores and return a list of scores."""
    rng = np.random.RandomState(seed)
    labels = np.array(labels)
    scores = np.array(scores)

    bootstrapped_scores = []
    for _ in range(n_bootstraps):
        # Sample with replacement
        indices = rng.randint(0, len(labels), len(labels))
        if len(np.unique(labels[indices])) < 2:
            continue  # Skip iteration if only one class is present
        score = roc_auc_score(labels[indices], scores[indices])
        bootstrapped_scores.append(score)

    return bootstrapped_scores

def bootstrap_enrichment_factors(y_true, y_score, fpr_levels=[0.01, 0.02, 0.05], n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    n = len(y_true)
    ef_bootstrap = {fpr: [] for fpr in fpr_levels}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_resample = y_true[idx]
        score_resample = y_score[idx]

        try:
            ef = compute_ef(y_resample, score_resample, fpr_levels)
            for fpr in fpr_levels:
                ef_bootstrap[fpr].append(ef[fpr])
        except ValueError:
            # In case one class is missing in resample
            continue

    return ef_bootstrap

def bootstrap_nef(y_true, y_score, fpr_levels=[0.01, 0.02, 0.05], n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    n = len(y_true)
    nef_bootstrap = {fpr: [] for fpr in fpr_levels}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_resample = y_true[idx]
        score_resample = y_score[idx]

        try:
            nef = compute_nef(y_resample, score_resample, fpr_levels)
            for fpr in fpr_levels:
                nef_bootstrap[fpr].append(nef[fpr])
        except ValueError:
            # In case one class is missing in resample
            continue

    return nef_bootstrap

def plot_bootstrapped_metrics(target_metrics, save_path='bootstrapped_metrics.png', method='mean', output_dir='results_DUD_E'):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    def extract_metric_data(key_mean, key_ci=None, target_list=None):
        y_vals, lower, upper, valid_targets = [], [], [], []
        for t in target_list:
            m = target_metrics.get(t, {})
            if key_mean in m:
                y = m[key_mean]
                ci = m.get(key_ci, (y, y)) if key_ci else (y, y)
                y_vals.append(y)
                lower.append(max(y - ci[0], 0.0))
                upper.append(max(ci[1] - y, 0.0))
                valid_targets.append(t)
            else:
                y_vals.append(None)
                lower.append(0)
                upper.append(0)
                valid_targets.append(t)
        return y_vals, [lower, upper], valid_targets

    all_targets = list(target_metrics.keys())

    # --- Extract and sort NEF ---
    nef_key_mean = 'NEF (Affinity) 1.0% mean' if method == 'mean' else 'NEF (Affinity) 1.0% median'
    nef_key_ci = 'NEF (Affinity) 1.0% CI'
    nef_y, nef_err, nef_targets = extract_metric_data(nef_key_mean, nef_key_ci, all_targets)

    # Filter out None values for sorting
    nef_data = [(t, y, err) for t, y, err in zip(nef_targets, nef_y, zip(*nef_err)) if y is not None]
    nef_data_sorted = sorted(nef_data, key=lambda x: x[1], reverse=True)
    sorted_targets = [t for t, _, _ in nef_data_sorted]

    # Re-extract NEF data in sorted order (ensures alignment)
    nef_y_sorted, nef_err_sorted, _ = extract_metric_data(nef_key_mean, nef_key_ci, sorted_targets)

    # --- Extract EF ---
    ef_key_mean = 'EF (Affinity) 1.0% mean' if method == 'mean' else 'EF (Affinity) 1.0% median'
    ef_key_ci = 'EF (Affinity) 1.0% CI'
    ef_y, ef_err, _ = extract_metric_data(ef_key_mean, ef_key_ci, sorted_targets)

    # --- Extract ROC AUC ---
    auc_key_mean = 'boot_aff_mean' if method == 'mean' else 'boot_aff_median'
    auc_key_ci = 'boot_aff_ci'
    roc_y, roc_err, _ = extract_metric_data(auc_key_mean, auc_key_ci, sorted_targets)

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # --- Plot NEF ---
    axs[0].errorbar(range(len(sorted_targets)), nef_y_sorted, yerr=nef_err_sorted, fmt='o', ecolor='teal', color='teal', capsize=0)
    axs[0].set_ylabel("NEF 1%")
    axs[0].set_ylim(0, 1)
    axs[0].set_title("Dense (Affinity)")

    # --- Plot EF ---
    axs[1].errorbar(range(len(sorted_targets)), ef_y, yerr=ef_err, fmt='o', ecolor='teal', color='teal', capsize=0)
    axs[1].set_ylabel("EF 1%")
    axs[1].set_ylim(0, 100)

    # --- Plot AUC ---
    axs[2].errorbar(range(len(sorted_targets)), roc_y, yerr=roc_err, fmt='o', ecolor='teal', color='teal', capsize=0)
    axs[2].set_ylabel("AUC")
    axs[2].set_ylim(0, 1)
    axs[2].axhline(0.5, color='black', linestyle='--', linewidth=1)

    axs[-1].set_xticks(range(len(sorted_targets)))
    axs[-1].set_xticklabels(sorted_targets, rotation=90, fontsize=8)

    for ax in axs:
        ax.grid(False)

    plt.tight_layout()
    full_save_path = os.path.join(output_dir, save_path)
    plt.savefig(full_save_path, dpi=300)
    plt.close()
    print(f"Saved bootstrapped metrics plot to {full_save_path}")

def plot_roc_and_distributions(target_metrics,
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

    num_targets = len(target_metrics)
    cols = 3
    rows = (num_targets + cols - 1) // cols

    fig_roc_pose, axs_roc_pose = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    fig_roc_affinity, axs_roc_affinity = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    fig_dist, axs_dist = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))

    axs_roc_pose = axs_roc_pose.flatten()
    axs_roc_affinity = axs_roc_affinity.flatten()
    axs_dist = axs_dist.flatten()

    for idx, (target, data) in enumerate(target_metrics.items()):
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

    print(f"Saved pose-based ROC curves to: {os.path.join(output_dir, roc_outfile)}")
    print(f"Saved affinity-based ROC curves to: {os.path.join(output_dir, affinity_roc_outfile)}")
    print(f"Saved affinity distributions to: {os.path.join(output_dir, dist_outfile)}")

def plot_ef_nef_grouped_bar(target_metrics, output_dir='results_DUD_E'):
    # Ensure the output directory exists
    output_dir = ensure_output_dir(output_dir)
    sns.set_theme(style="whitegrid")

    # Extract target names
    targets = list(target_metrics.keys())

    # Extract EF and NEF values for each level
    ef_levels = ['EF (Affinity) 1.0% mean', 'EF (Affinity) 2.0% mean', 'EF (Affinity) 5.0% mean']
    nef_levels = ['NEF (Affinity) 1.0% mean', 'NEF (Affinity) 2.0% mean', 'NEF (Affinity) 5.0% mean']

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
