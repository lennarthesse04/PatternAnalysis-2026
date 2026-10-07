import os
import glob
import torch
from torch.utils.data import Dataset
import nibabel as nib


class HipMRIDataset(Dataset):
    """
    Custom PyTorch Dataset for COMP3710 HipMRI 2D slices (.nii.gz format).
    Written entirely using PyTorch tensor operations (no NumPy).
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing .nii.gz slices 
                            (e.g., '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train')
            transform (callable, optional): Optional transform/augmentation to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform

        # Collect all .nii.gz files in the target directory
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))

        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No .nii.gz files found in {data_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        # Load NIfTI file and extract raw tensor directly
        nii_img = nib.load(file_path)
        
        # Convert directly to PyTorch tensor from buffer memory
        tensor_img = torch.as_tensor(nii_img.dataobj, dtype=torch.float32)

        # Remove singleton dimensions (e.g. shape [H, W, 1] -> [H, W])
        tensor_img = torch.squeeze(tensor_img)

        # Min-Max Normalization using PyTorch operations [0.0, 1.0]
        min_val = torch.min(tensor_img)
        max_val = torch.max(tensor_img)
        denom = max_val - min_val

        if denom > 0:
            tensor_img = (tensor_img - min_val) / denom
        else:
            tensor_img = torch.zeros_like(tensor_img)

        # Add Channel dimension -> shape: [1, H, W]
        tensor_img = tensor_img.unsqueeze(0)

        if self.transform:
            tensor_img = self.transform(tensor_img)

        return tensor_img