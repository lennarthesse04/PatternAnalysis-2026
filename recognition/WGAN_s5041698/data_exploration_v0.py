import os
from torch.utils.data import DataLoader
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
    print(f"Batch Tensor Shape: {batch.shape}")   # Output: [16, 1, H, W]
    print(f"Min Value: {batch.min():.2f}, Max Value: {batch.max():.2f}")
    break