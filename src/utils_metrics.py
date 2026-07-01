import numpy as np
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim


def mse_numpy(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean((a - b) ** 2))


def ssim_numpy(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    return float(
        ssim(
            a,
            b,
            data_range=max(float(b.max() - b.min()), 1e-8),
        )
    )


def extract_simple_image_features(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)

    mean = np.mean(image)
    std = np.std(image)
    p10 = np.percentile(image, 10)
    p25 = np.percentile(image, 25)
    p50 = np.percentile(image, 50)
    p75 = np.percentile(image, 75)
    p90 = np.percentile(image, 90)

    return np.array([mean, std, p10, p25, p50, p75, p90], dtype=np.float32)


def frechet_distance_from_features(real_features: np.ndarray, generated_features: np.ndarray) -> float:
    real_features = np.asarray(real_features, dtype=np.float64)
    generated_features = np.asarray(generated_features, dtype=np.float64)

    mu_real = np.mean(real_features, axis=0)
    mu_generated = np.mean(generated_features, axis=0)

    cov_real = np.cov(real_features, rowvar=False)
    cov_generated = np.cov(generated_features, rowvar=False)

    diff = mu_real - mu_generated

    cov_mean = sqrtm(cov_real.dot(cov_generated))

    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    distance = diff.dot(diff) + np.trace(cov_real + cov_generated - 2.0 * cov_mean)

    return float(distance)


def binary_classification_metrics(y_true, y_pred) -> dict:
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }