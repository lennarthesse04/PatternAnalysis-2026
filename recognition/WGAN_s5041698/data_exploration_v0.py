import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import umap
from dataset import HipMRIDataset

BASE_PATH = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"

train_dir = os.path.join(BASE_PATH, "keras_slices_train")
val_dir   = os.path.join(BASE_PATH, "keras_slices_validate")
test_dir  = os.path.join(BASE_PATH, "keras_slices_test")

# Create PyTorch Datasets
train_dataset = HipMRIDataset(data_dir=train_dir)
val_dataset   = HipMRIDataset(data_dir=val_dir)
test_dataset  = HipMRIDataset(data_dir=test_dir)

# Create DataLoaders
BATCH_SIZE = 128

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Sanity Check
for batch in train_loader:
    print(f"Batch Tensor Shape: {batch.shape}")   # Output: [128, 1, H, W]
    print(f"Min Value: {batch.min():.2f}, Max Value: {batch.max():.2f}")
    break

# ==========================================
# UMAP DATA AUDIT EXTENSION
# ==========================================

print("\nCollecting samples for UMAP data audit...")
max_samples = 2000  # Adjust based on memory; 2000-5000 is plenty for a 2D map
collected_images = []
total_count = 0

for batch in train_loader:
    imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
    collected_images.append(imgs.detach().cpu().numpy())
    total_count += imgs.shape[0]
    if total_count >= max_samples:
        break

# Concatenate and flatten: [N, 1, H, W] -> [N, H * W]
X_real = np.concatenate(collected_images, axis=0)[:max_samples]
N, C, H, W = X_real.shape
X_flat = X_real.reshape(N, C * H * W)

print(f"Flattened shape for auditing: {X_flat.shape}")

# Step 1: PCA to reduce noise and dimensionality (e.g., to 50 dimensions)
print("Running PCA...")
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_flat)

# Step 2: UMAP projection to 2D
print("Running UMAP...")
umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
embedding_2d = umap_reducer.fit_transform(X_pca)

# Step 3: Plot the 2D Real Data Baseline
plt.figure(figsize=(8, 6))
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.5, s=10, c='royalblue')
plt.title("WGAN Data Audit: Real Hip MRI 2D UMAP Baseline")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("real_mri_umap_audit.png", dpi=300, bbox_inches='tight')
plt.show()
print("Saved baseline UMAP plot to 'real_mri_umap_audit.png'")