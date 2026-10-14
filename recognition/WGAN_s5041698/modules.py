import torch
import torch.nn as nn
import torch.nn.functional as F

class WGANGP(nn.Module):
    """
    Wasserstein GAN with Gradient Penalty (WGAN-GP) for Synthetic Medical Slices.
    Contains both the Generator and Critic (Discriminator) networks, along with 
    the gradient penalty loss formulation.
    """
    def __init__(self, latent_dim=128, img_channels=1, feature_map_size=64):
        super(WGANGP, self).__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        
        # Initialize Generator and Critic
        self.generator = Generator(latent_dim=latent_dim, img_channels=img_channels, base_filters=feature_map_size)
        self.critic = Critic(img_channels=img_channels, base_filters=feature_map_size)

    def forward(self, z):
        """Generates fake images from latent vectors z."""
        return self.generator(z)

    def compute_gradient_penalty(self, real_samples, fake_samples, device):
        """
        Computes the gradient penalty loss for WGAN-GP.
        Enforces the 1-Lipschitz constraint by penalizing critic gradient norms.
        """
        batch_size = real_samples.size(0)
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        alpha = alpha.expand_as(real_samples)

        # Get interpolated samples
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)

        # Calculate critic output on interpolated images
        critic_interpolated = self.critic(interpolated)

        # Calculate gradients of critic with respect to interpolated images
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty


class Generator(nn.Module):
    """
    Generator network using resize-convolutions to eliminate 
    checkerboard artifacts common in transposed convolutions.
    """
    def __init__(self, latent_dim=128, img_channels=1, base_filters=64):
        super(Generator, self).__init__()
        
        # Maps latent vector into starting feature map dimensions (e.g., 4x4 or 8x8)
        self.init_size = 8 # Assuming target image is 256x256 (8 -> 16 -> 32 -> 64 -> 128 -> 256)
        self.l1 = nn.Sequential(nn.Linear(latent_dim, base_filters * 16 * self.init_size * self.init_size))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(base_filters * 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Upsample to 16x16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_filters * 16, base_filters * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Upsample to 32x32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_filters * 8, base_filters * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Upsample to 64x64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_filters * 4, base_filters * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Upsample to 128x128
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_filters * 2, base_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.LeakyReLU(0.2, inplace=True),

            # Upsample to 256x256 & Output Image Layer
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_filters, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output range [-1, 1] matching medical normalization
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Critic(nn.Module):
    """
    Critic (Discriminator) network using LayerNormalization instead of BatchNorm 
    to prevent inter-sample dependencies required by WGAN-GP gradient penalty.
    """
    def __init__(self, img_channels=1, base_filters=64):
        super(Critic, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1)]
            if bn:
                # LayerNorm is preferred over BatchNorm in WGAN-GP critics
                block.append(nn.LayerNorm([out_filters, 64, 64])) # Dynamic sizing handled downstream if resolution changes
            block.append(nn.LeakyReLU(0.2, inplace=True))
            block.append(nn.Dropout2d(0.25))
            return block

        self.model = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(img_channels, base_filters, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 -> 64
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 -> 32
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 -> 16
            nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 16 -> 8
            nn.Conv2d(base_filters * 8, base_filters * 16, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters * 16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(base_filters * 16 * 8 * 8, 1) # Assumes 256x256 input spatial dimensions
        )

    def forward(self, img):
        validity = self.model(img)
        return validity