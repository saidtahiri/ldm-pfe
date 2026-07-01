import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.ldm_config import (
    CLASSIFIER_BATCH_SIZE,
    CLASSIFIER_DIR,
    CLASSIFIER_EPOCHS,
    CLASSIFIER_LR,
    CLASSIF_DIR,
    METRICS_DIR,
    SEED,
)
from src.ldm_dataset import get_simple_classifier_loader
from src.utils_common import get_device, set_seed
from src.utils_metrics import binary_classification_metrics


class SimpleCNNClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def evaluate(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    return binary_classification_metrics(y_true, y_pred)


def main():
    set_seed(SEED)
    device = get_device()
    print(f"Périphérique utilisé : {device}")

    train_loader = get_simple_classifier_loader(
        real_split="train",
        synthetic_manifest_path=None,
        batch_size=CLASSIFIER_BATCH_SIZE,
        shuffle=True,
    )

    test_loader = get_simple_classifier_loader(
        real_split="test",
        synthetic_manifest_path=None,
        batch_size=CLASSIFIER_BATCH_SIZE,
        shuffle=False,
    )

    model = SimpleCNNClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CLASSIFIER_LR)

    metrics_rows = []
    best_f1 = -1.0

    for epoch in range(1, CLASSIFIER_EPOCHS + 1):
        model.train()
        loss_sum = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        metrics = evaluate(model, test_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = loss_sum / max(len(train_loader), 1)
        metrics["scenario"] = "S1_real_only"

        metrics_rows.append(metrics)

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                },
                CLASSIFIER_DIR / "classifier_s1_best.pt",
            )

        print(
            f"S1 epoch {epoch}/{CLASSIFIER_EPOCHS} | "
            f"loss={metrics['train_loss']:.6f} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"recall={metrics['recall']:.4f} | "
            f"f1={metrics['f1_score']:.4f}"
        )

    df = pd.DataFrame(metrics_rows)
    df.to_csv(METRICS_DIR / "classifier_s1_metrics.csv", index=False)

    final_metrics = metrics_rows[-1]
    confusion = pd.DataFrame(
        [
            {
                "scenario": "S1_real_only",
                "tn": final_metrics["tn"],
                "fp": final_metrics["fp"],
                "fn": final_metrics["fn"],
                "tp": final_metrics["tp"],
            }
        ]
    )

    confusion.to_csv(CLASSIF_DIR / "confusion_matrix_s1.csv", index=False)

    print("Entraînement classifieur S1 terminé.")


if __name__ == "__main__":
    main()