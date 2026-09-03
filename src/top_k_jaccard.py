import numpy as np

from itertools import combinations

def top_k_jaccard_all(predictions, k):
    scores = []

    for run_a, run_b in combinations(
        range(predictions.shape[1]), 2
    ):
        top_a = set(np.argsort(predictions[:, run_a])[-k:])
        top_b = set(np.argsort(predictions[:, run_b])[-k:])

        union = top_a | top_b

        jaccard = (
            len(top_a & top_b) / len(union)
            if union else 0
        )

        scores.append(jaccard)
    return np.array(scores)