import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.ldm_config import (
    AE_DIR,
    MAX_QUALITY_EVAL_IMAGES,
    METRICS_DIR,
    RECON_DIR,
    SEED,
)
from src.ldm_dataset import get_dataloader
from src.models_autoencoder import AutoencoderKL
from src.utils_common import get_device, set_seed
from src.utils_metrics import (
    extract_simple_image_features,
    frechet_distance_from_features,
    mse_numpy,
    ssim_numpy,
)


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

    return model


def save_single_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(array, 0.0, 1.0)
    image_uint8 = (array * 255.0).astype("uint8")
    Image.fromarray(image_uint8, mode="L").save(path)


def load_generated_features() -> np.ndarray:
    manifest_path = METRICS_DIR / "generated_samples_manifest.csv"

    if not manifest_path.exists():
        return np.empty((0, 7), dtype=np.float32)

    manifest = pd.read_csv(manifest_path)
    features = []

    for _, row in manifest.iterrows():
        path = Path(row["image_path"])

        if not path.exists():
            continue

        image = Image.open(path).convert("L")
        image = image.resize((128, 128))
        array = np.asarray(image).astype(np.float32) / 255.0
        features.append(extract_simple_image_features(array))

    if len(features) == 0:
        return np.empty((0, 7), dtype=np.float32)

    return np.stack(features, axis=0)


def main() -> None:
    set_seed(SEED)
    device = get_device()
    print(f"Périphérique utilisé : {device}")

    model = load_autoencoder(device)

    test_loader = get_dataloader(
        split="test",
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    rows = []
    real_features = []

    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            if index >= MAX_QUALITY_EVAL_IMAGES:
                break

            images = batch["image"].to(device)
            labels = batch["label"]

            reconstruction, _, _ = model(images)

            original_np = images[0, 0].detach().cpu().numpy()
            reconstruction_np = reconstruction[0, 0].detach().cpu().numpy()

            mse_value = mse_numpy(original_np, reconstruction_np)
            ssim_value = ssim_numpy(original_np, reconstruction_np)

            real_features.append(extract_simple_image_features(original_np))

            recon_path = RECON_DIR / f"quality_reconstruction_{index + 1:05d}.png"
            save_single_image(reconstruction_np, recon_path)

            rows.append(
                {
                    "image_index": index + 1,
                    "label": int(labels[0].item()),
                    "mse_reconstruction": mse_value,
                    "ssim_reconstruction": ssim_value,
                    "reconstruction_path": str(recon_path),
                }
            )

    metrics_df = pd.DataFrame(rows)
    metrics_path = METRICS_DIR / "image_quality_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    summary_rows = []

    if len(rows) > 0:
        summary_rows.append(
            {
                "metric": "mse_reconstruction_mean",
                "value": float(metrics_df["mse_reconstruction"].mean()),
            }
        )
        summary_rows.append(
            {
                "metric": "ssim_reconstruction_mean",
                "value": float(metrics_df["ssim_reconstruction"].mean()),
            }
        )

    generated_features = load_generated_features()

    if len(real_features) > 1 and generated_features.shape[0] > 1:
        real_features_array = np.stack(real_features, axis=0)
        frechet_simple = frechet_distance_from_features(real_features_array, generated_features)

        summary_rows.append(
            {
                "metric": "frechet_simple_feature_distance",
                "value": frechet_simple,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = METRICS_DIR / "image_quality_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Métriques qualité sauvegardées : {metrics_path}")
    print(f"Résumé qualité sauvegardé : {summary_path}")


if __name__ == "__main__":
    main()