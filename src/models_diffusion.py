import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2

        exponent = torch.arange(half_dim, device=timesteps.device).float()
        exponent = -math.log(10000.0) * exponent / max(half_dim - 1, 1)

        frequencies = torch.exp(exponent)
        args = timesteps.float()[:, None] * frequencies[None, :]

        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return embedding


class TimeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)

        self.time_proj = nn.Linear(time_dim, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        time_term = self.time_proj(time_emb)[:, :, None, None]
        h = h + time_term

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.skip(x)


class LatentUNet(nn.Module):
    def __init__(
        self,
        latent_channels: int = 3,
        time_dim: int = 128,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.class_embedding = nn.Embedding(num_classes, time_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.down1 = TimeBlock(latent_channels, 64, time_dim)
        self.down2 = TimeBlock(64, 128, time_dim)
        self.down3 = TimeBlock(128, 256, time_dim)

        self.pool = nn.AvgPool2d(2)

        self.mid = TimeBlock(256, 256, time_dim)

        self.up2 = TimeBlock(256 + 128, 128, time_dim)
        self.up1 = TimeBlock(128 + 64, 64, time_dim)

        self.out = nn.Conv2d(64, latent_channels, kernel_size=3, padding=1)

    def forward(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_emb = self.time_embedding(timesteps)

        if labels is not None:
            labels = labels.long()
            t_emb = t_emb + self.class_embedding(labels)

        t_emb = self.time_mlp(t_emb)

        h1 = self.down1(z_t, t_emb)
        h2 = self.down2(self.pool(h1), t_emb)
        h3 = self.down3(self.pool(h2), t_emb)

        h_mid = self.mid(h3, t_emb)

        h = F.interpolate(h_mid, size=h2.shape[-2:], mode="nearest")
        h = torch.cat([h, h2], dim=1)
        h = self.up2(h, t_emb)

        h = F.interpolate(h, size=h1.shape[-2:], mode="nearest")
        h = torch.cat([h, h1], dim=1)
        h = self.up1(h, t_emb)

        return self.out(h)


class DDPMScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.device = device

        self.betas = torch.linspace(
            beta_start,
            beta_end,
            num_train_timesteps,
            device=device,
        )

        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timesteps]
        alpha_bar = alpha_bar.view(-1, 1, 1, 1)

        noisy_samples = (
            torch.sqrt(alpha_bar) * original_samples
            +
            torch.sqrt(1.0 - alpha_bar) * noise
        )

        return noisy_samples

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        prev_timestep: int,
    ) -> torch.Tensor:
        alpha_bar_t = self.alpha_bars[timestep]

        if prev_timestep >= 0:
            alpha_bar_prev = self.alpha_bars[prev_timestep]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=sample.device)

        pred_original = (
            sample - torch.sqrt(1.0 - alpha_bar_t) * model_output
        ) / torch.sqrt(alpha_bar_t)

        direction = torch.sqrt(1.0 - alpha_bar_prev) * model_output
        prev_sample = torch.sqrt(alpha_bar_prev) * pred_original + direction

        return prev_sample