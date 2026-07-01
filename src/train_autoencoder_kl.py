import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.ldm_config import (
    AE_BATCH_SIZE,
    AE_BETA_KL,
    AE_DIR,
    AE_EPOCHS,
    AE_LR,
    FIGURES_DIR,
    METRICS_DIR,
    RECON_DIR,
    SEED,
)
from src.ldm_dataset import get_dataloader
from src.models_autoencoder import AutoencoderKL, kl_loss, reconstruction_l1_loss
from src.utils_common import append_row_to_csv, get_device, save_image_grid, save_metric_curve, set_seed


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

    train_loader = get_dataloader(
        split="train",
        batch_size=AE_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = get_dataloader(
        split="val",
        batch_size=AE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = AutoencoderKL(latent_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR)

    train_metrics_path = METRICS_DIR / "autoencoder_train_metrics.csv"
    val_metrics_path = METRICS_DIR / "autoencoder_val_metrics.csv"

    if train_metrics_path.exists():
        train_metrics_path.unlink()

    if val_metrics_path.exists():
        val_metrics_path.unlink()

    best_val_loss = float("inf")

    for epoch in range(1, AE_EPOCHS + 1):
        model.train()

        total_loss_sum = 0.0
        rec_loss_sum = 0.0
        kl_loss_sum = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)

            optimizer.zero_grad(set_to_none=True)

            reconstruction, z_mu, z_logvar = model(images)

            rec = reconstruction_l1_loss(images, reconstruction)
            kl = kl_loss(z_mu, z_logvar)
            total = rec + AE_BETA_KL * kl

            total.backward()
            optimizer.step()

            total_loss_sum += total.item()
            rec_loss_sum += rec.item()
            kl_loss_sum += kl.item()

        n_train = max(len(train_loader), 1)

        train_row = {
            "epoch": epoch,
            "loss": total_loss_sum / n_train,
            "reconstruction_l1": rec_loss_sum / n_train,
            "kl": kl_loss_sum / n_train,
            "beta_kl": AE_BETA_KL,
        }

        append_row_to_csv(train_metrics_path, train_row)

        model.eval()
        val_total_sum = 0.0
        val_rec_sum = 0.0
        val_kl_sum = 0.0

        with torch.no_grad():
            last_val_images = None
            last_val_recon = None

            for batch in val_loader:
                images = batch["image"].to(device)

                reconstruction, z_mu, z_logvar = model(images)

                rec = reconstruction_l1_loss(images, reconstruction)
                kl = kl_loss(z_mu, z_logvar)
                total = rec + AE_BETA_KL * kl

                val_total_sum += total.item()
                val_rec_sum += rec.item()
                val_kl_sum += kl.item()

                last_val_images = images
                last_val_recon = reconstruction

        n_val = max(len(val_loader), 1)
        val_loss = val_total_sum / n_val

        val_row = {
            "epoch": epoch,
            "loss": val_loss,
            "reconstruction_l1": val_rec_sum / n_val,
            "kl": val_kl_sum / n_val,
            "beta_kl": AE_BETA_KL,
        }

        append_row_to_csv(val_metrics_path, val_row)

        save_checkpoint(
            AE_DIR / "autoencoder_kl_last.pt",
            model,
            optimizer,
            epoch,
            val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                AE_DIR / "autoencoder_kl_best.pt",
                model,
                optimizer,
                epoch,
                val_loss,
            )

        if last_val_images is not None and last_val_recon is not None:
            grid = torch.cat(
                [
                    last_val_images[:4].detach().cpu(),
                    last_val_recon[:4].detach().cpu(),
                ],
                dim=0,
            )

            save_image_grid(
                grid,
                RECON_DIR / f"reconstruction_epoch_{epoch:03d}.png",
                title=f"AutoencoderKL - epoch {epoch}",
                max_images=8,
            )

        print(
            f"Epoch {epoch}/{AE_EPOCHS} | "
            f"train_loss={train_row['loss']:.6f} | "
            f"val_loss={val_row['loss']:.6f} | "
            f"val_rec={val_row['reconstruction_l1']:.6f} | "
            f"val_kl={val_row['kl']:.6f}"
        )

    save_metric_curve(
        train_metrics_path,
        columns=["loss", "reconstruction_l1", "kl"],
        output_path=FIGURES_DIR / "autoencoder_train_curves.png",
        title="Courbes d'entraînement de l'AutoencoderKL",
    )

    save_metric_curve(
        val_metrics_path,
        columns=["loss", "reconstruction_l1", "kl"],
        output_path=FIGURES_DIR / "autoencoder_val_curves.png",
        title="Courbes de validation de l'AutoencoderKL",
    )

    print("Entraînement AutoencoderKL terminé.")


if __name__ == "__main__":
    main()