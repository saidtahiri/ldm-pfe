import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.ldm_config import FIGURES_DIR, METRICS_DIR


def load_last_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = pd.read_csv(path)
    row = df.iloc[-1].to_dict()
    return row


def main():
    s1 = load_last_metrics(METRICS_DIR / "classifier_s1_metrics.csv")
    s2 = load_last_metrics(METRICS_DIR / "classifier_s2_metrics.csv")

    rows = [
        {
            "scenario": "S1_real_only",
            "accuracy": s1["accuracy"],
            "precision": s1["precision"],
            "recall": s1["recall"],
            "f1_score": s1["f1_score"],
            "tn": s1["tn"],
            "fp": s1["fp"],
            "fn": s1["fn"],
            "tp": s1["tp"],
        },
        {
            "scenario": "S2_real_plus_ldm",
            "accuracy": s2["accuracy"],
            "precision": s2["precision"],
            "recall": s2["recall"],
            "f1_score": s2["f1_score"],
            "tn": s2["tn"],
            "fp": s2["fp"],
            "fn": s2["fn"],
            "tp": s2["tp"],
        },
    ]

    df = pd.DataFrame(rows)
    comparison_path = METRICS_DIR / "classification_comparison.csv"
    df.to_csv(comparison_path, index=False)

    metrics = ["accuracy", "precision", "recall", "f1_score"]

    plt.figure(figsize=(8, 5))

    x = range(len(metrics))
    s1_values = [s1[m] for m in metrics]
    s2_values = [s2[m] for m in metrics]

    plt.bar([i - 0.2 for i in x], s1_values, width=0.4, label="S1 réel seul")
    plt.bar([i + 0.2 for i in x], s2_values, width=0.4, label="S2 réel + LDM")

    plt.xticks(list(x), metrics)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Comparaison diagnostique S1 vs S2")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()

    output_path = FIGURES_DIR / "s1_vs_s2_barplot.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Comparaison sauvegardée : {comparison_path}")
    print(f"Figure sauvegardée : {output_path}")


if __name__ == "__main__":
    main()