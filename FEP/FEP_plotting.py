import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, kendalltau, spearmanr
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
import numpy as np

# === RESULTS FOLDER === #
RESULTS_DIR = "FEP/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_boxplot_with_jitter(FEP_data, output_file="boxplot_with_jitter.png"):
    """
    Generate a boxplot with overlaid jittered points showing affinity differences
    to experimental values, based on Dense, Def2018, and Valsson et al. 2025 predictions.
    """

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, output_file)

    data = []
    for target, ligands in FEP_data.items():
        for values in ligands.values():
            if 'exp_value' in values:
                exp = values['exp_value']

                if 'affinity_scores_Dense' in values and values['affinity_scores_Dense'] is not None:
                    data.append({
                        'Target': target,
                        'Difference (kcal/mol)': values['affinity_scores_Dense'] - exp,
                        'Type': 'Dense'
                    })

                if 'affinity_scores_Def2018' in values and values['affinity_scores_Def2018'] is not None:
                    data.append({
                        'Target': target,
                        'Difference (kcal/mol)': values['affinity_scores_Def2018'] - exp,
                        'Type': 'Default'
                    })

                if 'pred_value' in values and values['pred_value'] is not None:
                    data.append({
                        'Target': target,
                        'Difference (kcal/mol)': values['pred_value'] - exp,
                        'Type': 'Valsson et al. 2025'
                    })

    df = pd.DataFrame(data)

    plt.figure(figsize=(20, 8))

    # Define consistent colors for boxplots
    palette = {
        'Dense': '#66c2a5',
        'Default': '#fc8d62',
        'Valsson et al. 2025': '#8da0cb'
    }

    # Boxplot
    ax = sns.boxplot(x='Target', y='Difference (kcal/mol)', hue='Type', data=df,
                     showfliers=False, palette=palette)

    # Stripplot with fixed grey color
    sns.stripplot(x='Target', y='Difference (kcal/mol)', hue='Type', data=df,
                  dodge=True, marker='o', alpha=0.1, size=6, palette='dark:gray', legend=False)

    # Custom legend handles using boxplot colors
    custom_handles = [Patch(facecolor=palette[name], label=name) for name in palette]
    ax.legend(handles=custom_handles, title='Model')

    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title('Affinity Differences to Experimental Values per Target')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()

def plot_kde_affinity_differences(FEP_data, output_file="kde_plots_per_target.png"):
    RESULTS_DIR = "FEP/results"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, output_file)

    # Prepare data
    data = []
    for target, ligands in FEP_data.items():
        for ligand, values in ligands.items():
            if 'exp_value' in values:
                exp_value = values['exp_value']

                if 'affinity_scores_Dense' in values and values['affinity_scores_Dense'] is not None:
                    diff_dense_exp = values['affinity_scores_Dense'] - exp_value
                    data.append({'Target': target, 'Difference': diff_dense_exp, 'Type': 'Dense'})

                if 'affinity_scores_Def2018' in values and values['affinity_scores_Def2018'] is not None:
                    diff_def2018_exp = values['affinity_scores_Def2018'] - exp_value
                    data.append({'Target': target, 'Difference': diff_def2018_exp, 'Type': 'Default'})

                if 'pred_value' in values and values['pred_value'] is not None:
                    diff_fep_exp = values['pred_value'] - exp_value
                    data.append({'Target': target, 'Difference': diff_fep_exp, 'Type': 'Valsson et al. 2025'})

    df = pd.DataFrame(data)

    # Plot setup
    plt.figure(figsize=(24, 12))

    # Manually plot each 'Type'
    for t in df['Type'].unique():
        subset = df[df['Type'] == t]
        sns.kdeplot(data=subset, x='Difference', fill=True, alpha=0.3, label=t)

    plt.axvline(0, color='gray', linestyle='--', linewidth=2)

    # Font sizes
    #plt.title('Kernel Density Estimate of Affinity Differences to Experimental Values', fontsize=36, weight='bold')
    plt.xlabel('Difference (kcal/mol)', fontsize=28)
    plt.ylabel('Density', fontsize=28)
    plt.xticks(fontsize=22, rotation=45, ha='right')
    plt.yticks(fontsize=22)

    # Add proper legend
    legend = plt.legend(title='Model', title_fontsize=24, fontsize=20)
    legend.get_frame().set_linewidth(0.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau

def plot_overall_correlation(FEP_data, target_metrics, save_path='FEP/results'):
    """
    Generate a figure with subplots showing scatter plots of predicted vs experimental values
    for all models. Each subplot includes a y=x reference line.

    Args:
        FEP_data (dict): FEP data with experimental and predicted values
        target_metrics (dict): Metrics dictionary per target/model
        save_path (str): Path to save the output figure
    """
    os.makedirs(save_path, exist_ok=True)

    models = ['Def2018', 'Dense', 'AEV-PLIG']
    all_exp = {model: [] for model in models}
    all_pred = {model: [] for model in models}

    # Collect predictions and experimental values across all targets
    for target, metrics in target_metrics.items():
        for model in models:
            if model not in metrics:
                continue
            for ligand, data in FEP_data.get(target, {}).items():
                if 'exp_value' not in data:
                    continue
                exp = data['exp_value']
                pred = (
                    data.get('pred_value') if model == 'AEV-PLIG'
                    else data.get(f'affinity_scores_{model}')
                )
                if pred is not None and not (np.isnan(exp) or np.isnan(pred)):
                    all_exp[model].append(exp)
                    all_pred[model].append(pred)

    # Plot setup
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

        # Plot scatter
        ax.scatter(exp_vals, pred_vals, alpha=0.6, s=60, label='Predictions')
        ax.plot([exp_vals.min(), exp_vals.max()], [exp_vals.min(), exp_vals.max()], 'k--', label='Ideal: y = x')

        ax.set_title(f'{model}', fontsize=24)
        ax.set_xlabel('Experimental Affinity', fontsize=20)
        if i == 0:
            ax.set_ylabel('Predicted Affinity', fontsize=20)

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(fontsize=14)
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    out_path = os.path.join(save_path, 'correlation_plots_all_models.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved combined correlation plot to {out_path}")
