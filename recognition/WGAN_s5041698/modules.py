import torch
import torch.nn as nn

class WGANGP(nn.Module):
    def __init__(self, latent_dim=128, img_channels=1, feature_map_size=64, lambda_gp=10.0):
        super(WGANGP, self).__init__()
        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp
        
        # Instantiate Generator and Critic internally using the provided parameters
        self.generator = Generator(latent_dim=latent_dim, channels=img_channels, feature_map_size=feature_map_size)
        self.critic = Critic(channels=img_channels, feature_map_size=feature_map_size)

    def forward(self, batch_size, device):
        """Generates fake images from random noise."""
        z = torch.randn(batch_size, self.latent_dim, 1, 1, device=device)
        return self.generator(z)

    def compute_gradient_penalty(self, real_imgs, fake_imgs, device):
        """Calculates the gradient penalty for WGAN-GP enforcement."""
        batch_size = real_imgs.size(0)
        
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        alpha = alpha.expand_as(real_imgs)
        
        # Get interpolated images
        interpolated = (alpha * real_imgs + ((1 - alpha) * fake_imgs)).requires_grad_(True)
        
        # Pass interpolated images through the critic
        critic_interpolated = self.critic(interpolated)
        
        # Calculate gradients w.r.t. interpolated images
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(critic_interpolated.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp
        return gradient_penalty

    def compute_critic_loss(self, real_imgs, fake_imgs, device):
        """Computes the loss function for the Critic."""
        real_validity = self.critic(real_imgs)
        fake_validity = self.critic(fake_imgs.detach())
        gp = self.gradient_penalty(real_imgs, fake_imgs.detach(), device)
        
        # WGAN-GP objective
        critic_loss = fake_validity.mean() - real_validity.mean() + gp
        return critic_loss, gp

    def compute_generator_loss(self, fake_imgs):
        """Computes the loss function for the Generator."""
        fake_validity = self.critic(fake_imgs)
        
        # Generator objective
        generator_loss = -fake_validity.mean()
        return generator_loss

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=128, channels=1, feature_map_size=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # 1. Input: latent_dim x 1 x 1 -> 1024 x 4 x 4
            nn.ConvTranspose2d(latent_dim, feature_map_size * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_map_size * 16),
            nn.ReLU(inplace=True),
            
            # 2. 4 x 4 -> 8 x 8
            nn.ConvTranspose2d(feature_map_size * 16, feature_map_size * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.ReLU(inplace=True),
            
            # 3. 8 x 8 -> 16 x 16
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(inplace=True),
            
            # 4. 16 x 16 -> 32 x 32
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(inplace=True),
            
            # 5. 32 x 32 -> 64 x 64
            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(inplace=True),
            
            # 6. 64 x 64 -> 128 x 128
            nn.ConvTranspose2d(feature_map_size, feature_map_size // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size // 2),
            nn.ReLU(inplace=True),
            
            # 7. 128 x 128 -> 256 x 256
            nn.ConvTranspose2d(feature_map_size // 2, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh() # Outputs values mapped to the range [-1, 1]
        )

    def forward(self, z):
        # Automatically reshape 2D input [batch_size, latent_dim] to 4D [batch_size, latent_dim, 1, 1]
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)
        return self.net(z)


class Critic(nn.Module):
    def __init__(self, channels=1, feature_map_size=64):
        super(Critic, self).__init__()
        # Evaluates a 256x256 image and outputs a scalar score (Wasserstein distance estimate)
        # Note: WGAN-GP typically avoids BatchNorm in the Critic to prevent gradient penalty interference.
        self.net = nn.Sequential(
            # 1. Input: channels x 256 x 256 -> (feature_map_size) x 128 x 128
            nn.Conv2d(channels, feature_map_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 2. 128 x 128 -> 64 x 64
            nn.Conv2d(feature_map_size, feature_map_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feature_map_size * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 3. 64 x 64 -> 32 x 32
            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feature_map_size * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4. 32 x 32 -> 16 x 16
            nn.Conv2d(feature_map_size * 4, feature_map_size * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feature_map_size * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 5. 16 x 16 -> 8 x 8
            nn.Conv2d(feature_map_size * 8, feature_map_size * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feature_map_size * 16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 6. 8 x 8 -> 4 x 4
            nn.Conv2d(feature_map_size * 16, feature_map_size * 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feature_map_size * 32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 7. 4 x 4 -> 1 x 1 (Scalar output)
            nn.Conv2d(feature_map_size * 32, 1, kernel_size=4, stride=1, padding=0, bias=False)
        )

    def forward(self, img):
        output = self.net(img)
        return output.view(-1, 1)