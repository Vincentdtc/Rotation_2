import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch

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


    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, output_file)

    # Prepare data for the KDE plot
    data = []

    for target, ligands in FEP_data.items():
        for ligand, values in ligands.items():
            if 'exp_value' in values:
                exp_value = values['exp_value']

                # Dense model prediction
                if 'affinity_scores_Dense' in values and values['affinity_scores_Dense'] is not None:
                    diff_dense_exp = values['affinity_scores_Dense'] - exp_value
                    data.append({
                        'Target': target,
                        'Difference': diff_dense_exp,
                        'Type': 'Dense'
                    })

                # Def2018 model prediction
                if 'affinity_scores_Def2018' in values and values['affinity_scores_Def2018'] is not None:
                    diff_def2018_exp = values['affinity_scores_Def2018'] - exp_value
                    data.append({
                        'Target': target,
                        'Difference': diff_def2018_exp,
                        'Type': 'Default'
                    })

                # Valsson et al. 2025 benchmark
                if 'pred_value' in values and values['pred_value'] is not None:
                    diff_fep_exp = values['pred_value'] - exp_value
                    data.append({
                        'Target': target,
                        'Difference': diff_fep_exp,
                        'Type': 'Valsson et al. 2025'
                    })

    # Create a DataFrame for easier plotting and analysis
    df = pd.DataFrame(data)

    # Create the KDE plot
    plt.figure(figsize=(20, 8))
    sns.kdeplot(data=df, x='Difference', hue='Type', common_norm=False, fill=True, alpha=0.3)

    # Add a vertical line at x = 0 for reference
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.title('Kernel Density Estimate of Affinity Differences to Experimental Values')
    plt.xlabel('Difference (kcal/mol)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot with increased resolution (dpi)
    plt.savefig(output_path, dpi=600)
    plt.close()
