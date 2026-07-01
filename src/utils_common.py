import random
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def append_row_to_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(path, index=False)


def save_image_grid(
    images: torch.Tensor,
    path: Path,
    title: str,
    max_images: int = 8,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    images = images.detach().cpu().clamp(0.0, 1.0)
    n = min(images.shape[0], max_images)

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.5))

    if n == 1:
        axes = [axes]

    for i in range(n):
        image = images[i, 0].numpy()
        axes[i].imshow(image, cmap="gray")
        axes[i].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


""" def save_metric_curve(
    csv_path: Path,
    columns: List[str],
    output_path: Path,
    title: str,
    xlabel: str = "epoch",
) -> None:
    if not csv_path.exists():
        print(f"Fichier introuvable : {csv_path}")
        return

    df = pd.read_csv(csv_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for column in columns:
        if column in df.columns:
            plt.plot(df[xlabel], df[column], label=column)

    plt.xlabel(xlabel)
    plt.ylabel("valeur")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close() """


def save_metric_curve(
    csv_path: Path,
    columns: List[str],
    output_path: Path,
    title: str,
    xlabel: str = "epoch",
) -> None:
    if not csv_path.exists():
        print(f"Fichier introuvable : {csv_path}")
        return

    df = pd.read_csv(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Axe primaire (gauche) pour les colonnes sauf 'kl'
    ax2 = None
    for column in columns:
        if column in df.columns:
            if column == 'kl':
                # On crée l'axe secondaire seulement si 'kl' est présent
                if ax2 is None:
                    ax2 = ax1.twinx()
                    ax2.set_ylabel('kl divergence', color='green')
                ax2.plot(df[xlabel], df[column], label=column, color='green')
                ax2.tick_params(axis='y', labelcolor='green')
            else:
                ax1.plot(df[xlabel], df[column], label=column)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("valeur (loss / recon)")
    ax1.set_title(title)
    ax1.grid(True)

    # Fusion des légendes des deux axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = (ax2.get_legend_handles_labels() if ax2 else ([], []))
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()