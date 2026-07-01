import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.ldm_config import (
    AE_DIR,
    DDPM_TIMESTEPS,
    DIFF_BATCH_SIZE,
    DIFF_DIR,
    DIFF_EPOCHS,
    DIFF_LR,
    FIGURES_DIR,
    METRICS_DIR,
    SEED,
)
from src.ldm_dataset import get_dataloader
from src.models_autoencoder import AutoencoderKL
from src.models_diffusion import DDPMScheduler, LatentUNet
from src.utils_common import append_row_to_csv, get_device, save_metric_curve, set_seed


def load_autoencoder(device: torch.device) -> AutoencoderKL:
    checkpoint_path = AE_DIR / "autoencoder_kl_best.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint AutoencoderKL introuvable. "
            "Lance d'abord : python src/train_autoencoder_kl.py"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = AutoencoderKL(latent_channels=3).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model


def save_checkpoint(path: Path, model, optimizer, epoch: int, loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


def main() -> None:
    set_seed(SEED)
    device = get_device()
    print(f"Périphérique utilisé : {device}")

    autoencoder = load_autoencoder(device)

    diffusion_model = LatentUNet(latent_channels=3, num_classes=2).to(device)
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=DIFF_LR)
    scheduler = DDPMScheduler(num_train_timesteps=DDPM_TIMESTEPS, device=device)

    # --- NOUVEAU : LOGIQUE DE REPRISE AUTOMATIQUE ---
    checkpoint_path = DIFF_DIR / "latent_unet_last.pt"
    start_epoch = 1
    best_loss = float("inf")

    if checkpoint_path.exists():
        print(f"--- Reprise de l'entraînement depuis : {checkpoint_path} ---")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        diffusion_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("loss", float("inf"))
        print(f"--- Reprise à partir de l'époque {start_epoch} ---")
    else:
        print("--- Aucun checkpoint trouvé, début de l'entraînement à zéro ---")
    # ------------------------------------------------

    train_loader = get_dataloader(
        split="train",
        batch_size=DIFF_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    metrics_path = METRICS_DIR / "latent_diffusion_train_metrics.csv"
    # Note : Si le fichier existe déjà (reprise), on continue d'écrire dedans.

    for epoch in range(start_epoch, DIFF_EPOCHS + 1):
        diffusion_model.train()
        loss_sum = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long().view(-1)

            with torch.no_grad():
                z_mu, z_logvar = autoencoder.encode(images)
                std = torch.exp(0.5 * z_logvar)
                latents = z_mu + std * torch.randn_like(std)

            noise = torch.randn_like(latents)

            timesteps = torch.randint(
                low=0,
                high=DDPM_TIMESTEPS,
                size=(latents.shape[0],),
                device=device,
            ).long()

            noisy_latents = scheduler.add_noise(
                original_samples=latents,
                noise=noise,
                timesteps=timesteps,
            )

            predicted_noise = diffusion_model(noisy_latents, timesteps, labels)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        avg_loss = loss_sum / max(len(train_loader), 1)

        row = {"epoch": epoch, "mse_noise": avg_loss}
        append_row_to_csv(metrics_path, row)

        # Sauvegarde du dernier état
        save_checkpoint(
            DIFF_DIR / "latent_unet_last.pt",
            diffusion_model,
            optimizer,
            epoch,
            avg_loss,
        )

        # Sauvegarde du meilleur état
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                DIFF_DIR / "latent_unet_best.pt",
                diffusion_model,
                optimizer,
                epoch,
                avg_loss,
            )

        print(f"Epoch {epoch}/{DIFF_EPOCHS} | mse_noise={avg_loss:.6f}")

    save_metric_curve(
        metrics_path,
        columns=["mse_noise"],
        output_path=FIGURES_DIR / "latent_diffusion_noise_mse_curve.png",
        title="MSE de prédiction du bruit latent",
    )

    print("Entraînement du modèle de diffusion latente terminé.")


if __name__ == "__main__":
    main()