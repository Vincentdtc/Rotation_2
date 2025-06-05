import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import glob
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D

# === CONFIG === #
LATENT_DIR = "results_DUD_E"
MAX_PCA_DIM = 50

# === LOAD LATENT VECTORS FROM ALL TARGETS === #
all_X = []
all_targets = []
all_labels = []

print("[INFO] Loading latent data from all targets...")
for npz_file in sorted(glob.glob(os.path.join(LATENT_DIR, "*_latent.npz"))):
    target = os.path.basename(npz_file).replace("_latent.npz", "")
    data = np.load(npz_file)
    X = data["X"]
    y = data["y"]
    
    all_X.append(X)
    all_targets.extend([target] * len(X))
    all_labels.extend(y.tolist())

all_X = np.vstack(all_X)
all_labels = np.array(all_labels)
print(f"[INFO] Loaded {len(all_targets)} samples from {len(set(all_targets))} targets.")
print(f"[INFO] Latent space dimension: {all_X.shape[1]}")

# === PCA REDUCTION === #
pca_dim = min(all_X.shape[1], MAX_PCA_DIM)
print(f"[INFO] Applying PCA → {pca_dim} dims")
X_pca = PCA(n_components=pca_dim).fit_transform(all_X)

# === t-SNE === #
print("[INFO] Running t-SNE...")
X_tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42).fit_transform(X_pca)

# === MAPPINGS === #
unique_targets = sorted(set(all_targets))
target_to_color = {
    t: c for t, c in zip(unique_targets, plt.cm.tab20(np.linspace(0, 1, len(unique_targets))))
}
markers = {0: 'o', 1: '^'}  # Not relevant anymore for the legend, but kept for visual distinction

# === PLOT === #
plt.figure(figsize=(12, 9))

# Plot points using color per target (still uses different shapes internally)
for target in unique_targets:
    color = target_to_color[target]
    for label in [0, 1]:
        idxs = [i for i, t in enumerate(all_targets) if t == target and all_labels[i] == label]
        if idxs:
            plt.scatter(X_tsne[idxs, 0], X_tsne[idxs, 1],
                        color=color, marker=markers[label], alpha=0.7, s=20)

# === LEGEND === #
# Only target-color legend
color_legend = [
    Line2D([0], [0], marker='o', color='w', label=target,
           markerfacecolor=target_to_color[target], markersize=8) 
    for target in unique_targets
]

plt.legend(handles=color_legend, title="Target", loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)

plt.title("t-SNE of Latent Space\nColor = Target", fontsize=14)
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.tight_layout()

# === SAVE & SHOW === #
output_path = os.path.join(LATENT_DIR, "all_targets_tsne_by_target.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[INFO] Plot saved to: {output_path}")
plt.show()
