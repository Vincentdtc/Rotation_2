import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_auc_score

def load_json_data(path):
    with open(path, 'r') as f:
        return json.load(f)

# === Load JSON data ===
paths = {
    'roc_aff_dense': 'results_DUD_E/avg_roc_affinity_dense.json',
    'roc_pose_dense': 'results_DUD_E/avg_roc_pose_dense.json',
    'prc_aff_dense': 'results_DUD_E/avg_prc_affinity_dense.json',
    'prc_pose_dense': 'results_DUD_E/avg_prc_pose_dense.json',
    'roc_aff_def': 'results_DUD_E/avg_roc_affinity_def.json',
    'roc_pose_def': 'results_DUD_E/avg_roc_pose_def.json',
    'prc_aff_def': 'results_DUD_E/avg_prc_affinity_def.json',
    'prc_pose_def': 'results_DUD_E/avg_prc_pose_def.json',
}

data = {k: load_json_data(v) for k, v in paths.items()}
a = 0.8  # Transparency for the curves

# === Compute AUCs ===
aucs = {
    'roc_aff_dense': auc(data['roc_aff_dense']['fpr'], data['roc_aff_dense']['tpr']),
    'roc_pose_dense': auc(data['roc_pose_dense']['fpr'], data['roc_pose_dense']['tpr']),
    'prc_aff_dense': auc(data['prc_aff_dense']['recall'], data['prc_aff_dense']['precision']),
    'prc_pose_dense': auc(data['prc_pose_dense']['recall'], data['prc_pose_dense']['precision']),
    'roc_aff_def': auc(data['roc_aff_def']['fpr'], data['roc_aff_def']['tpr']),
    'roc_pose_def': auc(data['roc_pose_def']['fpr'], data['roc_pose_def']['tpr']),
    'prc_aff_def': auc(data['prc_aff_def']['recall'], data['prc_aff_def']['precision']),
    'prc_pose_def': auc(data['prc_pose_def']['recall'], data['prc_pose_def']['precision']),
}

# === Create figure ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# === Plot ROC Curves ===
ax1.plot(data['roc_aff_dense']['fpr'], data['roc_aff_dense']['tpr'],
         label=f'Dense Affinity  (AUC = 0.733)', alpha=a)
ax1.plot(data['roc_pose_dense']['fpr'], data['roc_pose_dense']['tpr'],
         label=f'Dense Pose      (AUC = 0.698)', alpha=a)
ax1.plot(data['roc_aff_def']['fpr'], data['roc_aff_def']['tpr'],
         label=f'Default Affinity (AUC = 0.706)', alpha=a)
ax1.plot(data['roc_pose_def']['fpr'], data['roc_pose_def']['tpr'],
         label=f'Default Pose     (AUC = 0.677)', alpha=a)
ax1.plot([0, 1], [0, 1], 'k--', label='Random           (AUC = 0.500)')

ax1.set_title('Average ROC Curves')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend(loc='lower right')
ax1.grid(False)

# === Plot PRC Curves ===

# Trim low-recall region for dense affinity
recall = np.array(data['prc_aff_dense']['recall'])
precision = np.array(data['prc_aff_dense']['precision'])
mask = recall > 0.001
recall_trimmed = recall[mask]
precision_trimmed = precision[mask]

ax2.plot(recall_trimmed, precision_trimmed,
         label=f'Dense Affinity   (AUC = {aucs["prc_aff_dense"]:.3f})', alpha=a)
ax2.plot(data['prc_pose_dense']['recall'], data['prc_pose_dense']['precision'],
         label=f'Dense Pose       (AUC = {aucs["prc_pose_dense"]:.3f})', alpha=a)
ax2.plot(data['prc_aff_def']['recall'], data['prc_aff_def']['precision'],
         label=f'Default Affinity (AUC = {aucs["prc_aff_def"]:.3f})', alpha=a)
ax2.plot(data['prc_pose_def']['recall'], data['prc_pose_def']['precision'],
         label=f'Default Pose     (AUC = {aucs["prc_pose_def"]:.3f})', alpha=a)
ax2.axhline(y=0.02, linestyle='--', color='k', label='Random           (AUC = 0.020)')

ax2.set_title('Average PRC Curves')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.legend(loc='upper right')
ax2.grid(False)

# === Finalize and Save ===
plt.tight_layout()
output_file = 'results_DUD_E/average_roc_prc_combined.png'
plt.savefig(output_file, dpi=300)
print(f"[INFO] Saved plot to {output_file}")
