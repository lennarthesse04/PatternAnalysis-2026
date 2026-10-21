import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from modules import ConvVAE
from dataset import HipMRIDataset


def train_vae(train_dataloader, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvVAE(in_channels=1, latent_dim=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training Loop
    print("\n--- Starting VAE Training ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, total_bce, total_kld = 0.0, 0.0, 0.0

        for batch in train_dataloader:
            # If your dataset returns (image, label), change this to: batch, _ = batch
            batch = batch.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(batch)
            loss, bce, kld = model.vae_loss_function(recon_batch, batch, mu, logvar)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()

        # Average per batch item for cleaner logging
        num_batches = len(train_dataloader)
        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Loss: {train_loss/num_batches:.2f} | "
              f"BCE: {total_bce/num_batches:.2f} | "
              f"KLD: {total_kld/num_batches:.2f}")

    # Save Model Weights safely
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "./checkpoints/oasis_vae.pth")
    print("[+] Model saved to ./checkpoints/oasis_vae.pth")


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor()  # Normalizes to [0, 1]
    ])

    train_dataset = HipMRIDataset(data_dir="./data/train", transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    train_vae(train_dataloader, epochs=50)