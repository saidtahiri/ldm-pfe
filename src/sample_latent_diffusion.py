import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.ldm_config import (
    AE_DIR,
    DDPM_INFERENCE_STEPS,
    DDPM_TIMESTEPS,
    DIFF_DIR,
    LATENT_SIZE,
    METRICS_DIR,
    NUM_SYNTHETIC_NORMAL,
    NUM_SYNTHETIC_PNEUMONIA,
    SAMPLES_DIR,
    SEED,
)
from src.models_autoencoder import AutoencoderKL
from src.models_diffusion import DDPMScheduler, LatentUNet
from src.utils_common import get_device, save_image_grid, set_seed


def load_models(device: torch.device):
    ae_checkpoint_path = AE_DIR / "autoencoder_kl_best.pt"
    diff_checkpoint_path = DIFF_DIR / "latent_unet_best.pt"

    if not ae_checkpoint_path.exists():
        raise FileNotFoundError("Autoencoder introuvable. Entraîne d'abord l'AutoencoderKL.")

    if not diff_checkpoint_path.exists():
        raise FileNotFoundError("Modèle de diffusion introuvable. Entraîne d'abord la diffusion latente.")

    autoencoder_checkpoint = torch.load(ae_checkpoint_path, map_location=device)
    diffusion_checkpoint = torch.load(diff_checkpoint_path, map_location=device)

    autoencoder = AutoencoderKL(latent_channels=3).to(device)
    autoencoder.load_state_dict(autoencoder_checkpoint["model_state_dict"])
    autoencoder.eval()

    diffusion_model = LatentUNet(latent_channels=3, num_classes=2).to(device)
    diffusion_model.load_state_dict(diffusion_checkpoint["model_state_dict"])
    diffusion_model.eval()

    return autoencoder, diffusion_model


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    image_tensor = image_tensor.detach().cpu().clamp(0.0, 1.0)
    image_array = (image_tensor[0].numpy() * 255.0).astype("uint8")
    return Image.fromarray(image_array, mode="L")


def build_inference_timesteps() -> list:
    timesteps_tensor = torch.linspace(
        DDPM_TIMESTEPS - 1,
        0,
        DDPM_INFERENCE_STEPS,
    ).long()

    timesteps = [int(x.item()) for x in timesteps_tensor]

    unique_timesteps = []
    for t in timesteps:
        if len(unique_timesteps) == 0 or t != unique_timesteps[-1]:
            unique_timesteps.append(t)

    return unique_timesteps


def generate_for_class(
    autoencoder: AutoencoderKL,
    diffusion_model: LatentUNet,
    scheduler: DDPMScheduler,
    device: torch.device,
    class_label: int,
    class_name: str,
    n_images: int,
) -> tuple:
    rows = []
    generated_images = []

    class_dir = SAMPLES_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    timesteps = build_inference_timesteps()
    batch_size = 4
    generated_count = 0

    with torch.no_grad():
        while generated_count < n_images:
            current_batch_size = min(batch_size, n_images - generated_count)

            latents = torch.randn(
                current_batch_size,
                3,
                LATENT_SIZE[0],
                LATENT_SIZE[1],
                device=device,
            )

            labels = torch.full(
                (current_batch_size,),
                class_label,
                device=device,
                dtype=torch.long,
            )

            for index, timestep in enumerate(timesteps):
                if index + 1 < len(timesteps):
                    prev_timestep = timesteps[index + 1]
                else:
                    prev_timestep = -1

                t_tensor = torch.full(
                    (current_batch_size,),
                    int(timestep),
                    device=device,
                    dtype=torch.long,
                )

                predicted_noise = diffusion_model(latents, t_tensor, labels)
                latents = scheduler.step(
                    model_output=predicted_noise,
                    timestep=int(timestep),
                    sample=latents,
                    prev_timestep=int(prev_timestep),
                )

            images = autoencoder.decode(latents).clamp(0.0, 1.0)

            for i in range(current_batch_size):
                image_index = generated_count + i + 1
                output_path = class_dir / f"{class_name.lower()}_synthetic_{image_index:05d}.png"

                pil_image = tensor_to_pil(images[i])
                pil_image.save(output_path)

                rows.append(
                    {
                        "image_path": str(output_path),
                        "label": class_label,
                        "label_name": class_name,
                        "seed": SEED,
                        "sampling_steps": len(timesteps),
                        "autoencoder_checkpoint": str(AE_DIR / "autoencoder_kl_best.pt"),
                        "diffusion_checkpoint": str(DIFF_DIR / "latent_unet_best.pt"),
                    }
                )

                if len(generated_images) < 8:
                    generated_images.append(images[i].detach().cpu())

            generated_count += current_batch_size
            print(f"{class_name} générées : {generated_count}/{n_images}")

    return rows, generated_images


def main() -> None:
    set_seed(SEED)
    device = get_device()
    print(f"Périphérique utilisé : {device}")

    autoencoder, diffusion_model = load_models(device)
    scheduler = DDPMScheduler(num_train_timesteps=DDPM_TIMESTEPS, device=device)

    all_rows = []
    grid_images = []

    normal_rows, normal_grid = generate_for_class(
        autoencoder=autoencoder,
        diffusion_model=diffusion_model,
        scheduler=scheduler,
        device=device,
        class_label=0,
        class_name="NORMAL",
        n_images=NUM_SYNTHETIC_NORMAL,
    )

    pneumonia_rows, pneumonia_grid = generate_for_class(
        autoencoder=autoencoder,
        diffusion_model=diffusion_model,
        scheduler=scheduler,
        device=device,
        class_label=1,
        class_name="PNEUMONIA",
        n_images=NUM_SYNTHETIC_PNEUMONIA,
    )

    all_rows.extend(normal_rows)
    all_rows.extend(pneumonia_rows)
    grid_images.extend(normal_grid)
    grid_images.extend(pneumonia_grid)

    manifest = pd.DataFrame(all_rows)
    manifest_path = METRICS_DIR / "generated_samples_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    if len(grid_images) > 0:
        grid = torch.stack(grid_images[:8], dim=0)
        save_image_grid(
            grid,
            SAMPLES_DIR / "ldm_grid.png",
            title="Images synthétiques générées par LDM",
            max_images=8,
        )

    print(f"Manifest sauvegardé : {manifest_path}")
    print("Génération terminée.")


if __name__ == "__main__":
    main()