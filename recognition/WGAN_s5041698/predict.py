import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as compute_ssim
from tqdm import tqdm
from dataset import HipMRIDataset
from modules import Generator

def load_generator(checkpoint_path, latent_dim=128, device="cuda"):
    """
    Loads the trained WGAN-GP generator from a saved checkpoint file.
    """
    print(f"[*] Loading generator weights from: {checkpoint_path}")
    generator = Generator(latent_dim=latent_dim, img_channels=1, base_filters=64).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])

    generator.eval()
    return generator

def evaluate_cohort_ssim(
    checkpoint_path, 
    test_data_dir="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test", 
    num_samples=500, 
    latent_dim=128, 
    batch_size=64
):
    """
    Generates synthetic medical slices and computes SSIM against reference test slices.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluating on device: {device}")

    # 1. Load Generator Model
    generator = load_generator(checkpoint_path, latent_dim=latent_dim, device=device)

    # 2. Load Reference Test Dataset
    test_dataset = HipMRIDataset(data_dir=test_data_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    print(f"[*] Reference test dataset loaded with {len(test_dataset)} samples.")

    # 3. Generate Synthetic Cohort
    print(f"[*] Generating synthetic cohort of {num_samples} samples...")
    synthetic_images_list = []
    
    with torch.no_grad():
        generated_count = 0
        while generated_count < num_samples:
            curr_batch_size = min(batch_size, num_samples - generated_count)
            z = torch.randn(curr_batch_size, latent_dim, device=device)
            fake_imgs = generator(z)
            
            # Rescale from generator output range [-1, 1] to [0, 1] for SSIM calculation
            fake_imgs = fake_imgs * 0.5 + 0.5
            synthetic_images_list.append(fake_imgs.cpu().numpy())
            generated_count += curr_batch_size

    # Concatenate all generated batches: shape [num_samples, 1, H, W]
    synthetic_cohort = np.concatenate(synthetic_images_list, axis=0)

    # 4. Gather Reference Real Slices
    print("[*] Collecting reference test slices...")
    real_images_list = []
    for real_imgs in test_loader:
        if isinstance(real_imgs, (tuple, list)):
            real_imgs = real_imgs[0]
        # Rescale real images to [0, 1] if they are in [-1, 1]
        # (Adjust this scaling if your dataset class already returns [0, 1] or [0, 255])
        real_imgs = real_imgs * 0.5 + 0.5
        real_images_list.append(real_imgs.numpy())
        if sum(len(x) for x in real_images_list) >= num_samples:
            break
            
    real_cohort = np.concatenate(real_images_list, axis=0)[:num_samples]

    # 5. Compute SSIM Between Synthetic Cohort and Reference Real Cohort
    print("[*] Computing Structural Similarity Index (SSIM)...")
    ssim_scores = []

    # Compare each synthetic slice against its corresponding real test slice index 
    # (or random pairing if preferred)
    # Instead of direct index pairing:
    for i in range(num_samples):
        synth_slice = synthetic_cohort[i, 0]
        
        # Find the real slice in the test set that has the highest pixel/structural similarity
        # (or lowest L2 distance) to this specific synthetic slice
        distances = [np.mean((real_cohort[j, 0] - synth_slice) ** 2) for j in range(len(real_cohort))]
        best_match_idx = np.argmin(distances)
        real_slice = real_cohort[best_match_idx, 0]
        
        score = compute_ssim(real_slice, synth_slice, data_range=1.0)
        ssim_scores.append(score)

    mean_ssim = np.mean(ssim_scores)
    std_ssim = np.std(ssim_scores)

    print("\n" + "="*40)
    print(f"SSIM Evaluation Results (n={num_samples})")
    print("="*40)
    print(f"Mean SSIM : {mean_ssim:.4f}")
    print(f"Std SSIM  : {std_ssim:.4f}")
    print(f"Target    : > 0.60 (Acceptance Criteria)")
    print("="*40)

    return mean_ssim, std_ssim

if __name__ == "__main__":
    # Point this to your latest saved checkpoint from training
    checkpoint_file = "./checkpoints/wgan_gp_epoch_45.pth"  # Update with your actual checkpoint path
    
    if os.path.exists(checkpoint_file):
        evaluate_cohort_ssim(checkpoint_path=checkpoint_file, num_samples=500)
    else:
        print(f"[!] Checkpoint not found at {checkpoint_file}. Please check your path.")