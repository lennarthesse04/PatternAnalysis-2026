import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as compute_ssim
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import HipMRIDataset
from modules import Generator

def load_generator(checkpoint_path, latent_dim=128, device="cuda", type="WGAN-GP"):
    """Loads the trained WGAN-GP generator from a saved checkpoint file."""
    print(f"[*] Loading generator weights from: {checkpoint_path}")
    if type == "WGAN-GP":
        generator = Generator(latent_dim=latent_dim, img_channels=1, base_filters=64).to(device)
    elif type == "VAE":
        from modules import ConvVAE
        generator = ConvVAE(in_channels=1, latent_dim=latent_dim).to(device)
    else:
        raise ValueError(f"Unsupported model type: {type}. Choose 'WGAN-GP' or 'VAE'.")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dict = checkpoint['generator_state_dict'] if 'generator_state_dict' in checkpoint else checkpoint
    generator.load_state_dict(state_dict)
    
    generator.eval()
    return generator

def generate_synthetic_cohort(generator, num_samples=500, latent_dim=128, batch_size=64, device="cuda"):
    """
    Unified helper method to generate synthetic medical slices and return them in [0, 1] range.
    """
    print(f"[*] Generating synthetic cohort of {num_samples} samples...")
    synthetic_images_list = []
    
    with torch.no_grad():
        generated_count = 0
        while generated_count < num_samples:
            curr_batch_size = min(batch_size, num_samples - generated_count)
            z = torch.randn(curr_batch_size, latent_dim, device=device)

            # Check if it's a VAE or a GAN Generator
            if hasattr(generator, "decode"):
                # VAE decoding step
                fake_imgs = generator.decode(z)
            else:
                # WGAN-GP / GAN forward step
                fake_imgs = generator(z)
            
            # Rescale from generator output range [-1, 1] to [0, 1]
            fake_imgs = fake_imgs * 0.5 + 0.5
            synthetic_images_list.append(fake_imgs.cpu().numpy())
            generated_count += curr_batch_size

    return np.concatenate(synthetic_images_list, axis=0)

def load_dataset_as_array(data_dir, max_samples=None):
    """Unified helper method to load a dataset directory into a clean numpy array [N, H, W] in [0, 1]."""
    print(f"[*] Loading dataset from {data_dir}...")
    dataset = HipMRIDataset(data_dir=data_dir)
    images_list = []
    
    limit = max_samples if max_samples else len(dataset)
    for i in tqdm(range(min(len(dataset), limit)), desc=f"Loading {os.path.basename(data_dir)}"):
        img = dataset[i]
        if isinstance(img, (tuple, list)):
            img = img[0]
        # Rescale from [-1, 1] to [0, 1]
        img_rescaled = img.numpy() * 0.5 + 0.5
        images_list.append(img_rescaled[0])  # Drop channel dim to get [H, W]
        
    return np.stack(images_list, axis=0)

def evaluate_cohort_ssim(
    generator, 
    test_data_dir="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test", 
    num_samples=100, 
    latent_dim=128, 
    device="cuda"
):
    """
    Computes SSIM between synthetic cohort and reference test slices 
    using SSIM-based maximization for structural nearest-neighbor matching.
    """
    # 1. Generate synthetic cohort and ensure [0, 1] range
    synthetic_cohort = generate_synthetic_cohort(generator, num_samples=num_samples, latent_dim=latent_dim, device=device)
    if synthetic_cohort.min() < 0.0 or synthetic_cohort.max() > 1.0:
        synthetic_cohort = synthetic_cohort * 0.5 + 0.5  # Rescale from [-1, 1] to [0, 1] if needed

    # 2. Load real test cohort and ensure [0, 1] range
    real_cohort = load_dataset_as_array(test_data_dir, max_samples=num_samples)
    if real_cohort.min() < 0.0 or real_cohort.max() > 1.0:
        real_cohort = real_cohort * 0.5 + 0.5  # Rescale from [-1, 1] to [0, 1] if needed

    print(f"synthetic_cohort shape: {synthetic_cohort.shape}, real_cohort shape: {real_cohort.shape}")

    print("[*] Computing SSIM using Structural (SSIM-maximized) Nearest-Neighbor Matching...")
    ssim_scores = []
    ssim_output_dir = "./plots/ssim_pairs"
    os.makedirs(ssim_output_dir, exist_ok=True)

    for i in tqdm(range(num_samples), desc="Calculating Optimized SSIM"):
        synth_slice = synthetic_cohort[i, 0] if synthetic_cohort.ndim == 4 else synthetic_cohort[i]
        
        best_score = -1.0
        best_real_slice = None
        # Find the real test slice that maximizes SSIM (true structural match)
        for j in range(len(real_cohort)):
            real_slice = real_cohort[j, 0] if real_cohort.ndim == 4 else real_cohort[j]
            if i == 0 and j == 0:
                print(f"synth_slice shape: {synth_slice.shape}, real_slice shape: {real_slice.shape}")
            
            # Compute SSIM directly against candidate real slice
            score = compute_ssim(real_slice, synth_slice, data_range=1.0)
            if score > best_score:
                best_score = score
                best_real_slice = real_slice
                
        ssim_scores.append(best_score)

        # Save the best matching synthetic/real pair for only the first five iterations.
        if i < 5:
            plt.imsave(
                os.path.join(ssim_output_dir, f"pair_{i + 1}_synthetic.png"),
                synth_slice,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            plt.imsave(
                os.path.join(ssim_output_dir, f"pair_{i + 1}_real_ssim_{best_score:.4f}.png"),
                best_real_slice,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )

    mean_ssim = np.mean(ssim_scores)
    std_ssim = np.std(ssim_scores)

    print("\n" + "="*40)
    print(f"Optimized SSIM Evaluation Results (n={num_samples})")
    print("="*40)
    print(f"Mean SSIM : {mean_ssim:.4f}")
    print(f"Std SSIM  : {std_ssim:.4f}")
    print(f"Target    : > 0.60 (Acceptance Criteria)")
    print("="*40)

    return mean_ssim, std_ssim


def audit_nearest_neighbors(
    generator,
    train_data_dir="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train",
    num_synthetic_samples=5,
    num_neighbors=5,
    latent_dim=128,
    output_path="./plots/memorization_audit_grid.png",
    device="cuda"
):
    """Audits synthetic images against training images for memorization and plots the neighbor grid."""
    synthetic_array = generate_synthetic_cohort(generator, num_samples=num_synthetic_samples, latent_dim=latent_dim, device=device)
    train_array = load_dataset_as_array(train_data_dir)

    # Flatten for fast L2 distance computation
    train_flat = train_array.reshape(train_array.shape[0], -1)
    synthetic_flat = synthetic_array.reshape(synthetic_array.shape[0], -1)

    print("[*] Computing nearest training neighbors...")
    audit_results = []

    for i in range(num_synthetic_samples):
        synth_vec = synthetic_flat[i]
        distances = np.linalg.norm(train_flat - synth_vec, axis=1)
        nearest_indices = np.argsort(distances)[:num_neighbors]
        nearest_distances = distances[nearest_indices]
        
        audit_results.append({
            'synthetic_img': synthetic_array[i, 0],
            'neighbor_imgs': [train_array[idx] for idx in nearest_indices],
            'distances': nearest_distances
        })
        print(f"  -> Synthetic Sample {i+1}: Closest Training Neighbor Distance = {nearest_distances[0]:.4f}")

    print(f"[*] Saving visual audit grid to {output_path}...")
    fig, axes = plt.subplots(num_synthetic_samples, num_neighbors + 1, figsize=(14, 3 * num_synthetic_samples))
    if num_synthetic_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, res in enumerate(audit_results):
        axes[i, 0].imshow(res['synthetic_img'], cmap='gray')
        axes[i, 0].set_title(f"Synthetic #{i+1}" if i == 0 else "", fontsize=10)
        axes[i, 0].axis('off')
        
        for spine in axes[i, 0].spines.values():
            spine.set_edgecolor('blue')
            spine.set_linewidth(2)

        for j, (n_img, dist) in enumerate(zip(res['neighbor_imgs'], res['distances'])):
            ax = axes[i, j + 1]
            ax.imshow(n_img, cmap='gray')
            ax.set_title(f"Neighbor {j+1}\n(Dist: {dist:.1f})" if i == 0 else f"Dist: {dist:.1f}", fontsize=8)
            ax.axis('off')

    plt.suptitle("Privacy & Memorization Audit: Synthetic Slices vs. Nearest Training Scans", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print("[*] Audit complete!")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # checkpoint_file = f"./checkpoints/wgan_pg_epoch_58.pth"
    # if os.path.exists(checkpoint_file):
    #     # Load generator once for both evaluation tasks
    #     generator = load_generator(checkpoint_file, device=device)
        
    #     # 1. Run SSIM Evaluation
    #     evaluate_cohort_ssim(generator, device=device)
        
    #     # 2. Run Memorization Nearest-Neighbor Audit & Plot Grid
    #     # audit_nearest_neighbors(generator, device=device)
    # else:
    #     print(f"[!] Checkpoint not found at {checkpoint_file}.")

    checkpoint_file_vae = f"./checkpoints/oasis_vae.pth"
    if os.path.exists(checkpoint_file_vae):
        # Load VAE generator for evaluation
        vae_generator = load_generator(checkpoint_file_vae, type="VAE", device=device, latent_dim=10)
        
        # Run SSIM Evaluation for VAE
        evaluate_cohort_ssim(vae_generator, device=device, latent_dim=10)

    else:
        print(f"[!] Checkpoint not found at {checkpoint_file_vae}.")
