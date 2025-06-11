def plot_boxplot_with_jitter_and_violin(FEP_data, output_file="boxplot_with_jitter_and_violin.png"):

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, output_file)

    # Prepare data for the plot
    data = []

    for target, ligands in FEP_data.items():
        for ligand, values in ligands.items():
            if 'exp_value' in values and 'affinity_scores' in values and 'pred_value' in values:
                exp_value = values['exp_value']
                predicted_affinity = values['affinity_scores']
                fep_benchmark_affinity = values['pred_value']
                
                if fep_benchmark_affinity is not None:
                    # Difference between experimental and predicted affinity
                    diff_exp_pred = predicted_affinity - exp_value
                    # Difference between FEP benchmark and experimental affinity
                    diff_fep_exp = fep_benchmark_affinity - exp_value
                    
                    # Append the results to the data list
                    data.append({'Target': target, 'Difference': diff_exp_pred, 'Type': 'Predicted - Experimental'})
                    data.append({'Target': target, 'Difference': diff_fep_exp, 'Type': 'Valsson et al. 2025 - Experimental'})

    # Create a DataFrame for easier plotting and analysis
    df = pd.DataFrame(data)

    # Create the plot
    plt.figure(figsize=(22, 8))

    # First, add the violin plot
    ax = sns.violinplot(x='Target', y='Difference', hue='Type', data=df, split=True, inner='quart', palette="Set2")

    # Overlay the boxplot with jittered points
    sns.boxplot(x='Target', y='Difference', hue='Type', data=df, showfliers=False, palette="Set2", width=0.4, fliersize=0)

    # Overlay jittered points
    sns.stripplot(x='Target', y='Difference', hue='Type', data=df, dodge=True, marker='o', alpha=0.6, palette='dark:black', size=6)

    # Add a horizontal line at y = 0 for reference
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)

    # Set titles and labels
    plt.title('Affinity Differences per Target (Experimental vs Predicted, and Experimental vs Valsson et al. 2025)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot with increased resolution (dpi)
    plt.savefig(output_path, dpi=600)
    plt.close()


def plot_violin_affinity_differences(FEP_data, output_file="violin_plots_per_target.png"):

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, output_file)

    # Prepare data for the violin plot
    data = []

    for target, ligands in FEP_data.items():
        for ligand, values in ligands.items():
            if 'exp_value' in values and 'affinity_scores' in values and 'pred_value' in values:
                exp_value = values['exp_value']
                predicted_affinity = values['affinity_scores']
                fep_benchmark_affinity = values['pred_value']
                
                if fep_benchmark_affinity is not None:
                    # Difference between experimental and predicted affinity
                    diff_exp_pred = predicted_affinity - exp_value
                    # Difference between FEP benchmark and experimental affinity
                    diff_fep_exp =  fep_benchmark_affinity- exp_value
                    
                    # Append the results to the data list
                    data.append({'Target': target, 'Difference': diff_exp_pred, 'Type': 'Predicted - Experimental'})
                    data.append({'Target': target, 'Difference': diff_fep_exp, 'Type': 'Valsson et al. 2025 - Experimental'})

    # Create a DataFrame for easier plotting and analysis
    df = pd.DataFrame(data)

    # Create the violin plot
    plt.figure(figsize=(16, 8))  # Increased figure size for better resolution
    ax = sns.violinplot(x='Target', y='Difference', hue='Type', data=df, split=True, inner='quartile')

    # Add a horizontal line at y = 0 for reference
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title('Affinity Differences per Target (Experimental vs Predicted, and Experimental vs Valsson et al. 2025)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot with increased resolution (dpi)
    plt.savefig(output_path, dpi=600)  # Increased dpi for better resolution
    plt.close()

# === PLOT: RMSD PER TARGET WITH STD === #
def plot_rmsd_per_target(target_metrics, output_file="RMSD_per_target_with_std.png"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, output_file)

    targets = list(target_metrics.keys())
    rmsds = [target_metrics[t]['RMSD'] for t in targets]
    stds = [target_metrics[t]['STD'] for t in targets]

    plt.figure(figsize=(12, 6))
    plt.bar(targets, rmsds, yerr=stds, capsize=5, color='skyblue', edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('RMSD')
    plt.title('RMSD per Target (with Std Dev)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# === COLLECT AFFINITY DATA === #
def collect_affinity_data(FEP_data):
    exp, pred, pose, labels = [], [], [], []

    for target, ligands in FEP_data.items():
        for data in ligands.values():
            if all(k in data for k in ['exp_value', 'affinity_scores', 'pose_scores']):
                exp.append(data['exp_value'])
                pred.append(data['affinity_scores'])
                pose.append(data['pose_scores'])
                labels.append(target)

    return map(np.array, (exp, pred, pose, labels))

# === PER-TARGET CORRELATION PLOTS === #
def save_per_target_correlation(FEP_data, output_dir='target_combined_plots'):
    full_output_dir = os.path.join(RESULTS_DIR, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)

    for target, ligands in FEP_data.items():
        exp, pred, pose = [], [], []

        for data in ligands.values():
            if all(k in data for k in ['exp_value', 'affinity_scores', 'pose_scores']):
                exp.append(data['exp_value'])
                pred.append(data['affinity_scores'])
                pose.append(data['pose_scores'])

        if len(exp) < 10:
            continue

        exp, pred, pose = map(np.array, (exp, pred, pose))

        # Create a figure with 4 subplots (2 columns, 2 rows)
        fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

        # --- Subplot 1: Predicted vs Experimental Affinity
        ax1 = axs[0, 0]
        ax1.scatter(exp, pred, color='dodgerblue', edgecolor='k', alpha=0.7, s=50)

        # Fit without forcing through (0, 0)
        m1, b1 = np.polyfit(exp, pred, 1)
        r1, _ = pearsonr(exp, pred)

        # Plot the fit with dashed line and increased line width
        ax1.plot(exp, m1 * exp + b1, 'r--', label='Fit', linewidth=2)
        ax1.set(xlabel="Experimental Affinity", ylabel="Predicted Affinity",
                title="Predicted vs Experimental Affinity")
        ax1.text(0.05, 0.95, f'Pearson r = {r1:.3f}', transform=ax1.transAxes,
                 fontsize=12, va='top', bbox=dict(facecolor='white', alpha=0.8))
        ax1.grid(True)

        # --- Subplot 2: Pose Score vs Predicted Affinity
        ax2 = axs[0, 1]
        ax2.scatter(pose, pred, color='mediumseagreen', edgecolor='k', alpha=0.7, s=50)

        # Fit without forcing through (0, 0)
        m2, b2 = np.polyfit(pose, pred, 1)
        r2, _ = pearsonr(pose, pred)

        # Plot the fit with dashed line and increased line width
        ax2.plot(pose, m2 * pose + b2, 'r--', label='Fit', linewidth=2)
        ax2.set(xlabel="Pose Score", title="Pose Score vs Predicted Affinity")
        ax2.text(0.05, 0.95, f'Pearson r = {r2:.3f}', transform=ax2.transAxes,
                 fontsize=12, va='top', bbox=dict(facecolor='white', alpha=0.8))
        ax2.grid(True)

        # --- Subplot 3: Pose Score vs Experimental Affinity (Correlation)
        ax3 = axs[1, 0]
        ax3.scatter(pose, exp, color='orange', edgecolor='k', alpha=0.7, s=50)

        # Fit without forcing through (0, 0)
        m3, b3 = np.polyfit(pose, exp, 1)
        r3, _ = pearsonr(pose, exp)

        # Plot the fit with dashed line and increased line width
        ax3.plot(pose, m3 * pose + b3, 'r--', label='Fit', linewidth=2)
        ax3.set(xlabel="Pose Score", ylabel="Experimental Affinity", title="Pose Score vs Experimental Affinity")
        ax3.text(0.05, 0.95, f'Pearson r = {r3:.3f}', transform=ax3.transAxes,
                 fontsize=12, va='top', bbox=dict(facecolor='white', alpha=0.8))
        ax3.grid(True)

        # --- Subplot 4: Predicted vs Experimental Affinity (Correlation)
        ax4 = axs[1, 1]
        ax4.scatter(pred, exp, color='purple', edgecolor='k', alpha=0.7, s=50)

        # Fit without forcing through (0, 0)
        m4, b4 = np.polyfit(pred, exp, 1)
        r4, _ = pearsonr(pred, exp)

        # Plot the fit with dashed line and increased line width
        ax4.plot(pred, m4 * pred + b4, 'r--', label='Fit', linewidth=2)
        ax4.set(xlabel="Predicted Affinity", ylabel="Experimental Affinity", title="Predicted vs Experimental Affinity")
        ax4.text(0.05, 0.95, f'Pearson r = {r4:.3f}', transform=ax4.transAxes,
                 fontsize=12, va='top', bbox=dict(facecolor='white', alpha=0.8))
        ax4.grid(True)

        # Adjust layout and save the figure
        fig.suptitle(f"Target: {target}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(full_output_dir, f"{target}.png"), dpi=300)
        plt.close()

# === MEAN + STD GROUPED BAR PLOT === #
def plot_grouped_affinity(FEP_data, output_file="Grouped_Affinity_Barplot_Mean_STD_Per_Target.png"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, output_file)

    labels, exp_means, pred_means, model_means = [], [], [], []
    exp_stds, pred_stds, model_stds = [], [], []

    for target in sorted(FEP_data.keys()):
        exp, pred, model = [], [], []
        for ligand in FEP_data[target].values():
            if all(k in ligand for k in ['exp_value', 'pred_value', 'affinity_scores']):
                exp.append(ligand['exp_value'])
                pred.append(ligand['pred_value'])
                model.append(ligand['affinity_scores'])

        if exp and pred and model:
            labels.append(target)
            exp_means.append(np.mean(exp)); exp_stds.append(np.std(exp))
            pred_means.append(np.mean(pred)); pred_stds.append(np.std(pred))
            model_means.append(np.mean(model)); model_stds.append(np.std(model))

    x = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, exp_means, width, yerr=exp_stds, label='Experimental', color='lightblue', capsize=5)
    ax.bar(x, pred_means, width, yerr=pred_stds, label='Valsson et al. 2025', color='orange', capsize=5)
    ax.bar(x + width, model_means, width, yerr=model_stds, label='Predicted', color='green', capsize=5)

    ax.set_ylabel('Mean Affinity (pKd)')
    ax.set_title('Mean Experimental vs Predicted Affinity per Target')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_combined_correlation(FEP_data, output_file='combined_correlation_plots.png'):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, output_file)

    targets = sorted(FEP_data.keys())
    cmap = plt.get_cmap('tab20', len(targets))
    colors = {t: cmap(i) for i, t in enumerate(targets)}

    all_exp, all_pred, all_pose = [], [], []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    handles, labels = [], []

    for target in targets:
        exp, pred, pose = [], [], []
        for d in FEP_data[target].values():
            if all(k in d for k in ['exp_value', 'affinity_scores', 'pose_scores']):
                exp.append(d['exp_value'])
                pred.append(d['affinity_scores'])
                pose.append(d['pose_scores'])

        if len(exp) < 2:
            continue

        exp = np.array(exp)
        pred = np.array(pred)
        pose = np.array(pose)

        all_exp.extend(exp)
        all_pred.extend(pred)
        all_pose.extend(pose)

        h = ax1.scatter(exp, pred, color=colors[target], edgecolor='k', alpha=0.7, s=50, label=target)
        ax2.scatter(pose, pred, color=colors[target], edgecolor='k', alpha=0.7, s=50)
        handles.append(h)
        labels.append(target)

    # Convert to NumPy arrays before numerical operations
    all_exp = np.array(all_exp)
    all_pred = np.array(all_pred)
    all_pose = np.array(all_pose)

    if len(all_exp) > 1:
        m1, b1 = np.polyfit(all_exp, all_pred, 1)
        r1, _ = pearsonr(all_exp, all_pred)
        ax1.plot(all_exp, m1 * all_exp + b1, 'r--')
        ax1.set_title(f"Predicted vs Experimental Affinity\n(r = {r1:.2f})")

    if len(all_pose) > 1:
        m2, b2 = np.polyfit(all_pose, all_pred, 1)
        r2, _ = pearsonr(all_pose, all_pred)
        ax2.plot(all_pose, m2 * all_pose + b2, 'r--')
        ax2.set_title(f"Pose Score vs Predicted Affinity\n(r = {r2:.2f})")

    ax1.set_xlabel("Experimental Affinity")
    ax1.set_ylabel("Predicted Affinity")
    ax1.grid(True)
    ax2.set_xlabel("Pose Score")
    ax2.grid(True)

    fig.suptitle("Combined Correlation Plots for All Targets", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return handles, labels

# === STANDALONE LEGEND SAVER === #
def save_legend(handles, labels, filename="target_legend.png", ncol=2):
    output_path = os.path.join(RESULTS_DIR, filename)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig = plt.figure(figsize=(ncol * 2.5, len(labels) * 0.25))
    fig.legend(handles, labels, loc='center', frameon=False, ncol=ncol,
               fontsize=8, title="Targets", title_fontsize=9)
    plt.axis('off')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

