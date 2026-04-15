import pandas as pd
import numpy as np

def top_k_indices(preds, k_ratio=0.10):
    n = len(preds)
    k = int(n * k_ratio)
    return set(np.argsort(preds)[-k:])  # top k indices

def topk_jaccard(csv_path, k_ratio=0.10):
    df = pd.read_csv(csv_path)

    ids = df.iloc[:, 0]  # optional (if you want actual IDs later)
    preds = df.iloc[:, 1:].values  # shape: (n_samples, n_runs)

    n_runs = preds.shape[1]

    # Baseline = first run
    baseline_preds = preds[:, 0]
    topk_baseline = top_k_indices(baseline_preds, k_ratio)

    jaccard_scores = []

    # Compare each run to baseline
    for i in range(1, n_runs):
        current_preds = preds[:, i]
        topk_current = top_k_indices(current_preds, k_ratio)

        intersection = len(topk_baseline & topk_current)
        union = len(topk_baseline | topk_current)

        jaccard = intersection / union if union != 0 else 0
        jaccard_scores.append(jaccard)

    # Average Jaccard
    avg_jaccard = np.mean(jaccard_scores)

    return avg_jaccard, jaccard_scores