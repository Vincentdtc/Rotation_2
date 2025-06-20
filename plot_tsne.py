import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from openTSNE import TSNE
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

# === CONFIGURATION === #
LATENT_DIR = "results_DUD_E"  # Directory containing latent .npz files
MAX_PCA_DIM = 50              # Maximum PCA components to keep
TSNE_JOBS = 16               # Number of parallel jobs for t-SNE
FIGSIZE = (8, 6)             # Default figure size for plots
alpha_active = 0.7           # Transparency for active compounds
alpha_decoy = 0.01           # Transparency for decoy compounds
group_file = "groups_reduced.xlsx"  # Excel file mapping targets to groups


def load_npz(npz_file):
    """
    Load latent vectors and labels from a compressed .npz file.
    
    Parameters:
        npz_file (str): Path to the .npz file.
    
    Returns:
        tuple: (target_name, latent_vectors, labels)
    """
    target = os.path.basename(npz_file).replace("_latent.npz", "")
    data = np.load(npz_file)
    return target, data["X"], data["y"]


def run_tsne_pipeline():
    """
    Main pipeline to run PCA + t-SNE on latent vectors and generate plots.
    Prints progress and timing information for each step.
    """
    print("[INFO] Starting t-SNE pipeline...")
    pipeline_start = time.time()

    # === LOAD ALL LATENT FILES IN PARALLEL === #
    npz_files = sorted(glob.glob(os.path.join(LATENT_DIR, "*_latent.npz")))
    print(f"[INFO] Found {len(npz_files)} latent files to load")
    load_start = time.time()
    with ThreadPoolExecutor(max_workers=TSNE_JOBS) as executor:
        results = list(executor.map(load_npz, npz_files))
    print(f"[INFO] Loaded all latent files in {time.time() - load_start:.2f} seconds")

    # Combine all latent vectors, targets, and labels into single arrays
    all_X, all_targets, all_labels = [], [], []
    for target, X, y in results:
        all_X.append(X)
        all_targets.extend([target] * len(X))
        all_labels.extend(y.tolist())

    all_X = np.vstack(all_X)
    all_labels = np.array(all_labels)
    all_targets_np = np.array(all_targets)
    print(f"[INFO] Combined data shape: {all_X.shape}")
    print(f"[INFO] Labels distribution:\n{pd.Series(all_labels).value_counts()}")

    # === PCA REDUCTION === #
    pca_start = time.time()
    pca_dim = min(all_X.shape[1], MAX_PCA_DIM)
    X_pca = PCA(n_components=pca_dim).fit_transform(all_X)
    print(f"[INFO] PCA done in {time.time() - pca_start:.2f} seconds")

    # === t-SNE COMPUTATION === #
    tsne_start = time.time()
    print(f"[INFO] Running t-SNE with n_jobs={TSNE_JOBS}...")
    X_tsne = TSNE(
        n_components=2,
        perplexity=30,
        initialization="pca",
        n_jobs=TSNE_JOBS,
        random_state=42
    ).fit(X_pca)
    print(f"[INFO] t-SNE completed in {time.time() - tsne_start:.2f} seconds")

    # === CREATE DATAFRAME FOR PLOTTING === #
    df = pd.DataFrame({
        "target": all_targets_np,
        "label": all_labels,
        "x": X_tsne[:, 0],
        "y": X_tsne[:, 1]
    })

    # === ASSIGN COLORS TO TARGETS === #
    unique_targets = sorted(df["target"].unique())
    target_to_color = {
        t: c for t, c in zip(unique_targets, plt.cm.tab20(np.linspace(0, 1, len(unique_targets))))
    }

    # === PREPARE BOOLEAN MASKS FOR LABELS === #
    mask_dict = {
        (target, label): (df["target"] == target) & (df["label"] == label)
        for target in unique_targets for label in [0, 1]
    }

    # === MAP TARGETS TO GROUPS === #
    group_df = pd.read_excel(group_file)
    target_to_group = {
        target: col
        for col in group_df.columns
        for target in group_df[col].dropna()
    }

    df["group"] = df["target"].map(target_to_group).fillna("Unknown")
    unique_groups = sorted(df["group"].unique())
    group_to_color = {
        g: c for g, c in zip(unique_groups, plt.cm.tab10(np.linspace(0, 1, len(unique_groups))))
    }

    # === PLOT t-SNE COLORED BY TARGET === #
    plot_tsne_by_target(df, unique_targets, target_to_color, mask_dict)

    # === PLOT INDIVIDUAL TARGET FIGURES AND GROUP PDFs === #
    generate_per_target_plots(df, unique_targets, target_to_color, mask_dict, target_to_group)

    # === PLOT GROUP-COLORED t-SNE === #
    plot_tsne_by_group(df, unique_groups, group_to_color)

    print(f"[INFO] Pipeline finished in {time.time() - pipeline_start:.2f} seconds")


def plot_tsne_by_target(df, unique_targets, target_to_color, mask_dict):
    """
    Plot t-SNE embedding colored by target.
    Saves figure and stores axis limits globally for consistent zooming.
    
    Parameters:
        df (pd.DataFrame): Dataframe containing t-SNE coords, targets, labels.
        unique_targets (list): List of unique target names.
        target_to_color (dict): Mapping of target to color.
        mask_dict (dict): Masks for target-label pairs.
    """
    print("[INFO] Plotting t-SNE colored by target...")
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Scatter points for each target and label with different markers and alpha
    for target in unique_targets:
        color = target_to_color[target]
        for label in [0, 1]:
            mask = mask_dict[(target, label)]
            alpha = alpha_active if label == 1 else alpha_decoy
            marker = '^' if label == 1 else 'o'
            ax.scatter(df.loc[mask, "x"], df.loc[mask, "y"],
                       color=color, marker=marker, alpha=alpha, s=20, edgecolors='none')

    ax.set_title("t-SNE of Latent Space")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(
        handles=[
            Line2D([0], [0], marker='o', color='w', label=target,
                   markerfacecolor=target_to_color[target], markersize=8)
            for target in unique_targets
        ],
        title="Target", loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9
    )
    plt.tight_layout()

    output_path = os.path.join(LATENT_DIR, "all_targets_tsne_by_target.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved target-colored t-SNE plot to: {output_path}")

    # Store limits globally for consistent zoom in individual plots
    global xlim, ylim
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    plt.close(fig)


def generate_per_target_plots(df, unique_targets, target_to_color, mask_dict, target_to_group):
    """
    Generate individual t-SNE plots per target and save grouped PDFs.
    
    Parameters:
        df (pd.DataFrame): Dataframe with t-SNE coords, targets, labels.
        unique_targets (list): List of unique target names.
        target_to_color (dict): Mapping of target to color.
        mask_dict (dict): Masks for target-label pairs.
        target_to_group (dict): Mapping from target to group name.
    """
    print("[INFO] Generating per-target plots and PDFs...")
    output_dir = os.path.join(LATENT_DIR, "per_target_tsne")
    os.makedirs(output_dir, exist_ok=True)
    group_to_figures = {}

    for target in unique_targets:
        group = target_to_group.get(target, "Unknown")
        group_dir = os.path.join(output_dir, group)
        os.makedirs(group_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=FIGSIZE)
        color = target_to_color[target]

        # Plot active and decoy points with different markers and alpha
        for label in [0, 1]:
            mask = mask_dict[(target, label)]
            alpha = alpha_active if label == 1 else alpha_decoy
            marker = '^' if label == 1 else 'o'
            ax.scatter(df.loc[mask, "x"], df.loc[mask, "y"],
                       color=color, marker=marker, alpha=alpha, s=20, edgecolors='none')

        # Use consistent axis limits from overall plot for zoom consistency
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"{target}")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.grid(True)
        ax.legend(
            handles=[
                Line2D([0], [0], marker='^', color='gray', label='Active', linestyle='None', markersize=8),
                Line2D([0], [0], marker='o', color='gray', label='Decoy', linestyle='None', markersize=8, alpha=0.4)
            ],
            loc='upper right'
        )

        out_file = os.path.join(group_dir, f"{target}_tsne.png")
        fig.savefig(out_file, dpi=300, bbox_inches='tight')
        group_to_figures.setdefault(group, []).append(fig)
        plt.close(fig)

    # Save PDFs containing all target plots per group
    for group, figures in group_to_figures.items():
        pdf_path = os.path.join(output_dir, f"{group}_multipanel.pdf")
        with PdfPages(pdf_path) as pdf:
            for fig in figures:
                pdf.savefig(fig)
        print(f"[INFO] Saved multi-panel PDF for group '{group}' to: {pdf_path}")


def plot_tsne_by_group(df, unique_groups, group_to_color):
    """
    Plot t-SNE embedding colored by group and save the figure.
    
    Parameters:
        df (pd.DataFrame): Dataframe with t-SNE coords, groups, and labels.
        unique_groups (list): List of unique group names.
        group_to_color (dict): Mapping of group to color.
    """
    print("[INFO] Plotting t-SNE colored by group...")
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Scatter points for each group and label with different markers and alpha
    for group in unique_groups:
        color = group_to_color[group]
        for label in [0, 1]:
            mask = (df["group"] == group) & (df["label"] == label)
            alpha = alpha_active if label == 1 else alpha_decoy
            marker = '^' if label == 1 else 'o'
            ax.scatter(df.loc[mask, "x"], df.loc[mask, "y"],
                       color=color, marker=marker, alpha=alpha, s=20, edgecolors='none')

    ax.set_title("t-SNE of Latent Space (Colored by Group)")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(
        handles=[
            Line2D([0], [0], marker='o', color='w', label=group,
                   markerfacecolor=group_to_color[group], markersize=8)
            for group in unique_groups
        ],
        title="Group", loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9
    )
    plt.tight_layout()

    output_path = os.path.join(LATENT_DIR, "all_targets_tsne_by_group.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Saved group-colored t-SNE plot to: {output_path}")


# === MAIN === #
if __name__ == "__main__":
    run_tsne_pipeline()
