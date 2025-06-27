import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# === Constants === #
RESULTS_DIR = "FEP/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_boxplot_with_jitter(FEP_data, output_file="boxplot_with_jitter.png"):
    """
    Generate a boxplot with jittered points showing affinity differences
    between predicted and experimental values, across different models.

    Args:
        FEP_data (dict): Dictionary containing FEP results per target and ligand.
        output_file (str): Filename for the saved boxplot image.
    """
    output_path = os.path.join(RESULTS_DIR, output_file)
    data = []

    # Collect differences for each model
    for target, ligands in FEP_data.items():
        for values in ligands.values():
            exp = values.get('exp_value')
            if exp is None:
                continue

            for model_key, label in [
                ('affinity_scores_Dense', 'Dense'),
                ('affinity_scores_Def2018', 'Default'),
                ('pred_value', 'Valsson et al. 2025')
            ]:
                pred = values.get(model_key)
                if pred is not None:
                    data.append({
                        'Target': target,
                        'Difference (kcal/mol)': pred - exp,
                        'Type': label
                    })

    df = pd.DataFrame(data)

    plt.figure(figsize=(20, 8))

    # Color palette for models
    palette = {
        'Dense': '#66c2a5',
        'Default': '#fc8d62',
        'Valsson et al. 2025': '#8da0cb'
    }

    # Boxplot without outliers
    ax = sns.boxplot(
        x='Target', y='Difference (kcal/mol)', hue='Type', data=df,
        showfliers=False, palette=palette
    )

    # Overlay jittered (strip) points
    sns.stripplot(
        x='Target', y='Difference (kcal/mol)', hue='Type', data=df,
        dodge=True, marker='o', alpha=0.1, size=6, palette='dark:gray', legend=False
    )

    # Custom legend
    custom_handles = [Patch(facecolor=palette[name], label=name) for name in palette]
    ax.legend(handles=custom_handles, title='Model')

    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title('Affinity Differences to Experimental Values per Target')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()


def plot_kde_affinity_differences(FEP_data, output_file="kde_plots_per_target.png"):
    """
    Generate KDE plots of affinity differences per model across all targets.

    Args:
        FEP_data (dict): Dictionary containing FEP results per target and ligand.
        output_file (str): Filename for the saved KDE plot image.
    """
    output_path = os.path.join(RESULTS_DIR, output_file)
    data = []

    # Collect differences per model
    for target, ligands in FEP_data.items():
        for values in ligands.values():
            exp = values.get('exp_value')
            if exp is None:
                continue

            for model_key, label in [
                ('affinity_scores_Dense', 'Dense'),
                ('affinity_scores_Def2018', 'Default'),
                ('pred_value', 'Valsson et al. 2025')
            ]:
                pred = values.get(model_key)
                if pred is not None:
                    data.append({
                        'Target': target,
                        'Difference': pred - exp,
                        'Type': label
                    })

    df = pd.DataFrame(data)

    plt.figure(figsize=(24, 12))

    # Plot KDEs per model
    for model_type in df['Type'].unique():
        subset = df[df['Type'] == model_type]
        sns.kdeplot(
            data=subset, x='Difference', fill=True, alpha=0.3, label=model_type
        )

    plt.axvline(0, color='gray', linestyle='--', linewidth=2)
    plt.xlabel('Difference (kcal/mol)', fontsize=28)
    plt.ylabel('Density', fontsize=28)
    plt.xticks(fontsize=22, rotation=45, ha='right')
    plt.yticks(fontsize=22)

    legend = plt.legend(title='Model', title_fontsize=24, fontsize=20)
    legend.get_frame().set_linewidth(0.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()


def plot_overall_correlation(FEP_data, target_metrics, save_path=RESULTS_DIR):
    """
    Generate scatter plots of predicted vs experimental affinity for each model,
    including an identity line (y = x).

    Args:
        FEP_data (dict): Dictionary with experimental and predicted affinities.
        target_metrics (dict): Dictionary of metrics per target/model.
        save_path (str): Output directory to save the plot.
    """
    os.makedirs(save_path, exist_ok=True)

    models = ['Def2018', 'Dense', 'AEV-PLIG']
    all_exp = {model: [] for model in models}
    all_pred = {model: [] for model in models}

    # Aggregate predictions and experimental values
    for target, metrics in target_metrics.items():
        for model in models:
            if model not in metrics:
                continue
            for ligand, values in FEP_data.get(target, {}).items():
                exp = values.get('exp_value')
                pred = (
                    values.get('pred_value') if model == 'AEV-PLIG'
                    else values.get(f'affinity_scores_{model}')
                )
                if exp is not None and pred is not None and not (np.isnan(exp) or np.isnan(pred)):
                    all_exp[model].append(exp)
                    all_pred[model].append(pred)

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharex=True, sharey=True)
    fig.suptitle("Predicted vs Experimental Affinity (All Models)", fontsize=28, fontweight='bold')

    for i, model in enumerate(models):
        exp_vals = np.array(all_exp[model])
        pred_vals = np.array(all_pred[model])
        ax = axes[i]

        if len(exp_vals) < 2:
            ax.set_title(f"{model} (insufficient data)", fontsize=20)
            ax.axis('off')
            continue

        # Scatter and reference line
        ax.scatter(exp_vals, pred_vals, alpha=0.6, s=60, label='Predictions')
        ax.plot([exp_vals.min(), exp_vals.max()], [exp_vals.min(), exp_vals.max()],
                'k--', label='Ideal: y = x')

        ax.set_title(model, fontsize=24)
        ax.set_xlabel('Experimental Affinity (kcal/mol)', fontsize=20)
        if i == 0:
            ax.set_ylabel('Predicted Affinity (kcal/mol)', fontsize=20)

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(fontsize=14)
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    out_path = os.path.join(save_path, 'correlation_plots_all_models.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved combined correlation plot to {out_path}")
