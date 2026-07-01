from pathlib import Path
from src.utils_common import save_metric_curve

# Configuration des chemins vers tes fichiers de métriques définitifs
metrics_dir = Path("outputs/metrics")
output_dir = Path("outputs/figures")

# Liste des graphiques à générer
tasks = [
    {
        "csv": metrics_dir / "autoencoder_train_metrics.csv",
        "out": output_dir / "autoencoder_train_curves.png",
        "title": "Courbes d'entraînement de l'AutoencoderKL",
        "cols": ["loss", "reconstruction_l1", "kl"]
    },
    {
        "csv": metrics_dir / "autoencoder_val_metrics.csv",
        "out": output_dir / "autoencoder_val_curves.png",
        "title": "Courbes de validation de l'AutoencoderKL",
        "cols": ["loss", "reconstruction_l1", "kl"]
    }
]

for task in tasks:
    if task["csv"].exists():
        print(f"Génération de : {task['out'].name}...")
        save_metric_curve(task["csv"], task["cols"], task["out"], task["title"])
    else:
        print(f"Erreur : {task['csv']} introuvable.")

print("Processus terminé.")