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
import pandas as pd
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

def get_data(code, predictions, method, top_n):
    """
    Extract data for a given input code from predictions based on the specified method.

    Parameters:
    - code (str): Identifier for the prediction entry.
    - predictions (dict): Dictionary containing prediction data.
    - method (str): 'max_aff', 'max_pose', or 'mean'.
    - top_n (int): Number of top entries to average in 'mean' mode.

    Returns:
    - label (float), pose (float), affinity (float), latent (np.array)
    """
    entry = predictions[code]
    labels = np.array(entry['labels'])
    poses = np.array(entry['pose_scores'])
    affinities = np.array(entry['affinity_scores'])

    if method == 'max_aff':
        idx = np.argmax(affinities)
        return labels[idx], poses[idx], affinities[idx]

    elif method == 'max_pose':
        idx = np.argmax(poses)
        return labels[idx], poses[idx], affinities[idx]

    elif method == 'mean':
        top_idxs = np.argsort(affinities)[-top_n:][::-1]
        return (
            np.mean(labels[top_idxs]),
            np.mean(poses[top_idxs]),
            np.mean(affinities[top_idxs]),
        )

    else:
        raise ValueError("Method must be 'max_aff', 'max_pose', or 'mean'.")

def compute_ef_nef(all_labels, all_affinities, fpr_levels):
    all_affinities = np.asarray(all_affinities)
    all_labels = np.asarray(all_labels)

    total = len(all_labels)
    num_actives = np.sum(all_labels)

    ef_dict = {}
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

        ef_dict[level] = ef
        nef_dict[level] = nef

    return ef_dict, nef_dict

def bootstrap_roc_auc(labels, scores, n_bootstraps=1000, seed=42):
    """Bootstrap ROC AUC scores with stratified sampling and return a list of scores."""
    rng = np.random.RandomState(seed)
    labels = np.array(labels)
    scores = np.array(scores)

    bootstrapped_scores = []

    # Separate indices by class
    class_0_indices = np.where(labels == 0)[0]
    class_1_indices = np.where(labels == 1)[0]

    n_class_0 = len(class_0_indices)
    n_class_1 = len(class_1_indices)

    for _ in range(n_bootstraps):
        # Sample with replacement within each class
        sampled_class_0 = rng.choice(class_0_indices, size=n_class_0, replace=True)
        sampled_class_1 = rng.choice(class_1_indices, size=n_class_1, replace=True)

        # Combine indices
        sampled_indices = np.concatenate([sampled_class_0, sampled_class_1])

        # Check if both classes are present in the sample
        if len(np.unique(labels[sampled_indices])) < 2:
            continue  # skip if sample doesn't have both classes

        # Calculate ROC AUC for this bootstrap sample
        score = roc_auc_score(labels[sampled_indices], scores[sampled_indices])
        bootstrapped_scores.append(score)

    return bootstrapped_scores

def bootstrap_ef_nef(y_true, y_score, fpr_levels=[0.01, 0.02, 0.05], n_bootstrap=1000, seed=42):
    """
    Perform bootstrapping to compute both EF and NEF at given FPR levels.

    Parameters:
    - y_true: array-like of ground truth labels (0 or 1)
    - y_score: array-like of predicted scores
    - fpr_levels: list of FPR thresholds at which to evaluate EF and NEF
    - n_bootstrap: number of bootstrap samples
    - seed: random seed for reproducibility

    Returns:
    - ef_bootstrap: dict mapping FPR level to list of EF values
    - nef_bootstrap: dict mapping FPR level to list of NEF values
    """
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # Separate indices by class
    class_0_indices = np.where(y_true == 0)[0]
    class_1_indices = np.where(y_true == 1)[0]

    n_class_0 = len(class_0_indices)
    n_class_1 = len(class_1_indices)

    ef_bootstrap = {fpr: [] for fpr in fpr_levels}
    nef_bootstrap = {fpr: [] for fpr in fpr_levels}

    for _ in range(n_bootstrap):
        # Stratified sampling with replacement within each class
        sampled_class_0 = rng.choice(class_0_indices, size=n_class_0, replace=True)
        sampled_class_1 = rng.choice(class_1_indices, size=n_class_1, replace=True)

        sampled_indices = np.concatenate([sampled_class_0, sampled_class_1])
        y_resample = y_true[sampled_indices]
        score_resample = y_score[sampled_indices]

        if len(np.unique(y_resample)) < 2:
            continue  # Skip if both classes not present

        try:
            ef, nef = compute_ef_nef(y_resample, score_resample, fpr_levels)
            for fpr in fpr_levels:
                ef_bootstrap[fpr].append(ef[fpr])
                nef_bootstrap[fpr].append(nef[fpr])
        except ValueError:
            continue  # Skip on error (e.g., no actives)

    return ef_bootstrap, nef_bootstrap

def plot_bootstrapped_metrics(target_metrics, save_path='bootstrapped_metrics.png',
                              method='mean', output_dir='results_DUD_E',
                              mode='Affinity', auc_mode='aff',
                              reference_file='reference_metrics.xlsx'):
    """
    Plot NEF, EF, and AUC metrics for a given mode ('Affinity' or 'Pose'),
    and overlay reference values from an Excel file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load reference data
    ref_df = pd.read_excel(reference_file)
    ref_df.set_index('Target', inplace=True)

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

    def extract_reference_data(metric_name, targets):
        ref_y, ref_lower, ref_upper = [], [], []
        for t in targets:
            try:
                cv = ref_df.loc[t, f"{mode} CV ({metric_name})"]
                lb = ref_df.loc[t, f"{mode} LB ({metric_name})"]
                ub = ref_df.loc[t, f"{mode} UB ({metric_name})"]
                ref_y.append(cv)
                ref_lower.append(cv - lb)
                ref_upper.append(ub - cv)
            except KeyError:
                ref_y.append(None)
                ref_lower.append(0)
                ref_upper.append(0)
        return ref_y, [ref_lower, ref_upper]

    all_targets = list(target_metrics.keys())

    # --- NEF Keys ---
    nef_key_mean = f'NEF ({mode}) 1.0%'
    nef_key_ci = f'NEF ({mode}) 1.0% CI'

    # --- EF Keys ---
    ef_key_mean = f'EF ({mode}) 1.0%'
    ef_key_ci = f'EF ({mode}) 1.0% CI'

    # --- AUC Keys ---
    auc_key_mean = f'boot_{auc_mode}_{method}'
    auc_key_ci = f'boot_{auc_mode}_ci'

    # --- Extract and sort NEF ---
    nef_y, nef_err, nef_targets = extract_metric_data(nef_key_mean, nef_key_ci, all_targets)
    nef_data = [(t, y, err) for t, y, err in zip(nef_targets, nef_y, zip(*nef_err)) if y is not None]
    nef_data_sorted = sorted(nef_data, key=lambda x: x[1], reverse=True)
    sorted_targets = [t for t, _, _ in nef_data_sorted]
    nef_y_sorted, nef_err_sorted, _ = extract_metric_data(nef_key_mean, nef_key_ci, sorted_targets)

    # --- EF and AUC ---
    ef_y, ef_err, _ = extract_metric_data(ef_key_mean, ef_key_ci, sorted_targets)
    roc_y, roc_err, _ = extract_metric_data(auc_key_mean, auc_key_ci, sorted_targets)

    # --- Reference data ---
    ref_nef_y, ref_nef_err = extract_reference_data('NEF1%', sorted_targets)
    ref_ef_y, ref_ef_err = extract_reference_data('EF1%', sorted_targets)
    ref_auc_y, ref_auc_err = extract_reference_data('AUC', sorted_targets)

    # --- Plot ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    color = 'teal' if mode.lower() == 'affinity' else 'purple'
    ref_color = 'gray'

    # NEF plot
    axs[0].errorbar(range(len(sorted_targets)), nef_y_sorted, yerr=nef_err_sorted,
                    fmt='o', ecolor=color, color=color, capsize=0, label='Bootstrapped')
    axs[0].errorbar(range(len(sorted_targets)), ref_nef_y, yerr=ref_nef_err,
                    fmt='s', ecolor=ref_color, color=ref_color, alpha=0.5, capsize=0, label='Sunseri & Koes 2021')
    axs[0].set_ylabel("NEF 1%")
    axs[0].set_ylim(0, 1)
    axs[0].set_title(f"Dense ({mode})")
    axs[0].legend()

    # EF plot
    axs[1].errorbar(range(len(sorted_targets)), ef_y, yerr=ef_err,
                    fmt='o', ecolor=color, color=color, capsize=0)
    axs[1].errorbar(range(len(sorted_targets)), ref_ef_y, yerr=ref_ef_err,
                    fmt='s', ecolor=ref_color, color=ref_color, alpha=0.5, capsize=0)
    axs[1].set_ylabel("EF 1%")
    axs[1].set_ylim(0, 100)

    # AUC plot
    axs[2].errorbar(range(len(sorted_targets)), roc_y, yerr=roc_err,
                    fmt='o', ecolor=color, color=color, capsize=0)
    axs[2].errorbar(range(len(sorted_targets)), ref_auc_y, yerr=ref_auc_err,
                    fmt='s', ecolor=ref_color, color=ref_color, alpha=0.5, capsize=0)
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
            ax_pose.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color='purple')
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
            ax_aff.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color='teal')
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

def plot_ef_nef_grouped_bar_with_ci(target_metrics, output_dir='results_DUD_E', mode='Affinity', levels=[1.0, 5.0, 10.0]):
    """
    Plots grouped bar charts for EF and NEF with confidence intervals for Affinity or Pose.

    Parameters:
    - target_metrics: dict of per-target metrics.
    - output_dir: directory to save plots.
    - mode: 'Affinity' or 'Pose'
    - levels: list of float values (e.g. [1.0, 5.0, 10.0]) for enrichment percentages.
    """
    output_dir = ensure_output_dir(output_dir)
    sns.set_theme(style="whitegrid")
    targets = list(target_metrics.keys())
    x = np.arange(len(targets))
    width = 0.25

    ef_data = {}
    ef_err = {}
    nef_data = {}
    nef_err = {}

    for level in levels:
        ef_key = f'EF ({mode}) {level:.1f}%'
        ef_ci_key = f'EF ({mode}) {level:.1f}% CI'
        nef_key = f'NEF ({mode}) {level:.1f}%'
        nef_ci_key = f'NEF ({mode}) {level:.1f}% CI'

        ef_data[level] = [target_metrics[t].get(ef_key, 0.0) for t in targets]
        nef_data[level] = [target_metrics[t].get(nef_key, 0.0) for t in targets]

        ef_ci_vals = [target_metrics[t].get(ef_ci_key, (v, v)) for t, v in zip(targets, ef_data[level])]
        nef_ci_vals = [target_metrics[t].get(nef_ci_key, (v, v)) for t, v in zip(targets, nef_data[level])]

        ef_err[level] = [
            [max(v - ci[0], 0.0) for v, ci in zip(ef_data[level], ef_ci_vals)],  # Lower
            [max(ci[1] - v, 0.0) for v, ci in zip(ef_data[level], ef_ci_vals)]   # Upper
        ]
        nef_err[level] = [
            [max(v - ci[0], 0.0) for v, ci in zip(nef_data[level], nef_ci_vals)],
            [max(ci[1] - v, 0.0) for v, ci in zip(nef_data[level], nef_ci_vals)]
        ]

    # === EF Plot === #
    fig_ef, ax_ef = plt.subplots(figsize=(max(10, len(targets) * 0.6), 6))
    for i, level in enumerate(levels):
        ax_ef.bar(x + i * width, ef_data[level], width, yerr=ef_err[level],
                  capsize=5, label=f'EF {level:.1f}%', alpha=0.8)

    ax_ef.set_ylabel('Enrichment Factor (EF)')
    ax_ef.set_title(f'EF at {", ".join([f"{l:.1f}%" for l in levels])} by Target ({mode})')
    ax_ef.set_xticks(x + width * (len(levels) - 1) / 2)
    ax_ef.set_xticklabels(targets, rotation=45, ha='right')
    ax_ef.legend()
    ax_ef.grid(True, axis='y', linestyle='--', alpha=0.6)
    fig_ef.tight_layout()
    ef_path = os.path.join(output_dir, f"grouped_ef_plot_{mode.lower()}.png")
    fig_ef.savefig(ef_path, dpi=300)

    # === NEF Plot === #
    fig_nef, ax_nef = plt.subplots(figsize=(max(10, len(targets) * 0.6), 6))
    for i, level in enumerate(levels):
        ax_nef.bar(x + i * width, nef_data[level], width, yerr=nef_err[level],
                   capsize=5, label=f'NEF {level:.1f}%', alpha=0.8)

    ax_nef.set_ylabel('Normalized Enrichment Factor (NEF)')
    ax_nef.set_title(f'NEF at {", ".join([f"{l:.1f}%" for l in levels])} by Target ({mode})')
    ax_nef.set_xticks(x + width * (len(levels) - 1) / 2)
    ax_nef.set_xticklabels(targets, rotation=45, ha='right')
    ax_nef.legend()
    ax_nef.grid(True, axis='y', linestyle='--', alpha=0.6)
    fig_nef.tight_layout()
    nef_path = os.path.join(output_dir, f"grouped_nef_plot_{mode.lower()}.png")
    fig_nef.savefig(nef_path, dpi=300)

    print(f"Saved grouped EF plot to: {ef_path}")
    print(f"Saved grouped NEF plot to: {nef_path}")
