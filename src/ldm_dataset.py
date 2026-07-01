import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from PIL import Image

from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Lambdad,
    LoadImaged,
    Resized,
    ScaleIntensityd,
    ToTensord,
)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.ldm_config import DATA_DIR, IMAGE_SIZE, METRICS_DIR, SEED
from src.utils_common import set_seed


def find_images(split: str) -> List[Path]:
    split_dir = DATA_DIR / split

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Le dossier {split_dir} est introuvable. "
            "Vérifie que le dataset est placé dans data/raw/chest_xray/."
        )

    patterns = ["*.jpeg", "*.jpg", "*.png"]
    image_paths: List[Path] = []

    for pattern in patterns:
        image_paths.extend(split_dir.rglob(pattern))

    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise RuntimeError(
            f"Aucune image trouvée dans {split_dir}. "
            "Vérifie les extensions et l'arborescence du dataset."
        )

    return image_paths


def infer_label_from_path(path: Path) -> int:
    upper_path = str(path).upper()

    if "PNEUMONIA" in upper_path:
        return 1

    if "NORMAL" in upper_path:
        return 0

    raise ValueError(
        f"Impossible de déduire la classe de l'image : {path}. "
        "Le chemin doit contenir NORMAL ou PNEUMONIA."
    )


def build_data_list(split: str) -> List[Dict[str, object]]:
    image_paths = find_images(split)

    data = [
        {
            "image": str(path),
            "label": infer_label_from_path(path),
            "path": str(path),
            "split": split,
        }
        for path in image_paths
    ]

    return data


def get_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Lambdad(
                keys=["image"],
                func=lambda x: x[0:1, :, :] if x.shape[0] > 1 else x,
            ),
            Resized(keys=["image"], spatial_size=IMAGE_SIZE),
            ScaleIntensityd(keys=["image"]),
            ToTensord(keys=["image", "label"]),
        ]
    )


def get_dataloader(
    split: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    data = build_data_list(split)
    transforms = get_transforms()
    dataset = Dataset(data=data, transform=transforms)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


class SimpleImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        image_size: tuple = IMAGE_SIZE,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(path).convert("L")
        image = image.resize(self.image_size)
        array = torch.tensor(list(image.getdata()), dtype=torch.float32)
        array = array.view(1, self.image_size[1], self.image_size[0]) / 255.0

        return {
            "image": array,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(path),
        }


def get_simple_classifier_loader(
    real_split: str,
    synthetic_manifest_path: Optional[Path],
    batch_size: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    real_data = build_data_list(real_split)

    image_paths = [Path(item["path"]) for item in real_data]
    labels = [int(item["label"]) for item in real_data]

    if synthetic_manifest_path is not None and synthetic_manifest_path.exists():
        manifest = pd.read_csv(synthetic_manifest_path)

        for _, row in manifest.iterrows():
            image_paths.append(Path(row["image_path"]))
            labels.append(int(row["label"]))

    dataset = SimpleImageDataset(image_paths=image_paths, labels=labels)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def summarize_dataset() -> pd.DataFrame:
    rows = []

    for split in ["train", "val", "test"]:
        data = build_data_list(split)

        n_normal = sum(item["label"] == 0 for item in data)
        n_pneumonia = sum(item["label"] == 1 for item in data)
        total = len(data)

        rows.append(
            {
                "split": split,
                "NORMAL": n_normal,
                "PNEUMONIA": n_pneumonia,
                "total": total,
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = METRICS_DIR / "dataset_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nRésumé du dataset :")
    print(summary)
    print(f"\nFichier sauvegardé : {summary_path}")

    return summary


def sanity_check_loader(batch_size: int = 4) -> None:
    loader = get_dataloader(
        split="train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    images = batch["image"]
    labels = batch["label"]

    print("\nSanity check du DataLoader")
    print(f"Forme des images : {tuple(images.shape)}")
    print(f"Forme des labels : {tuple(labels.shape)}")
    print(f"Type des images : {images.dtype}")
    print(f"Valeur min images : {images.min().item():.4f}")
    print(f"Valeur max images : {images.max().item():.4f}")
    print(f"Labels du batch : {labels.tolist()}")

    expected_channels = 1
    expected_height, expected_width = IMAGE_SIZE

    assert images.ndim == 4, "Les images doivent être de forme (B, C, H, W)."
    assert images.shape[1] == expected_channels, "Les images doivent avoir un seul canal."
    assert images.shape[2] == expected_height, "Hauteur incorrecte après redimensionnement."
    assert images.shape[3] == expected_width, "Largeur incorrecte après redimensionnement."
    assert images.min() >= 0.0, "Les intensités doivent être >= 0 après normalisation."
    assert images.max() <= 1.0, "Les intensités doivent être <= 1 après normalisation."

    print("\nDataLoader validé avec succès.")


if __name__ == "__main__":
    set_seed(SEED)
    summarize_dataset()
    sanity_check_loader(batch_size=4)