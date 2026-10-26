import os
import torch
import torch.optim as optim
from modules import ConvVAE
from dataset import HipMRIDataset


def train_vae(train_dataloader, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvVAE(in_channels=1, latent_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("\n--- Starting VAE Training ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, total_mse, total_kld = 0.0, 0.0, 0.0

        for batch in train_dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(batch)
            loss, mse, kld = model.vae_loss_function(recon_batch, batch, mu, logvar)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total_mse += mse.item()
            total_kld += kld.item()

        num_batches = len(train_dataloader)
        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Loss: {train_loss / num_batches:.2f} | "
              f"MSE: {total_mse / num_batches:.2f} | "
              f"KLD: {total_kld / num_batches:.2f}")

    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "./checkpoints/oasis_vae.pth")
    print("[+] Model saved to ./checkpoints/oasis_vae.pth")


if __name__ == "__main__":
    train_dataset = HipMRIDataset(data_dir="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    train_vae(train_dataloader, epochs=100)