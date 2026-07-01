import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoencoderKL(nn.Module):
    def __init__(self, latent_channels: int = 3) -> None:
        super().__init__()

        self.latent_channels = latent_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )

        self.to_mu = nn.Conv2d(128, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv2d(128, latent_channels, kernel_size=1)

        self.decoder_input = nn.Conv2d(latent_channels, 128, kernel_size=1)

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor):
        features = self.encoder(x)
        z_mu = self.to_mu(features)
        z_logvar = self.to_logvar(features)
        return z_mu, z_logvar

    def reparameterize(self, z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_input(z)
        x = self.decoder(x)
        return x

    def forward(self, x: torch.Tensor):
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)
        reconstruction = self.decode(z)
        return reconstruction, z_mu, z_logvar


def kl_loss(z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(
        torch.sum(
            z_mu.pow(2) + torch.exp(z_logvar) - z_logvar - 1.0,
            dim=[1, 2, 3],
        )
    )


def reconstruction_l1_loss(x: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(reconstruction, x)