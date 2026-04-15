import pandas as pd
import numpy as np

def compute_cii(csv_path, threshold=0.5):
    df = pd.read_csv(csv_path)

    preds = df.iloc[:, 1:].values  # (n_samples, n_runs)

    # Convert to binary labels
    labels = (preds >= threshold).astype(int)

    # Compute instability per individual
    # CII = proportion of runs where label != majority label
    majority_label = np.round(labels.mean(axis=1)).astype(int)

    disagreements = (labels != majority_label[:, None]).sum(axis=1)
    cii_individual = disagreements / labels.shape[1]

    # Overall CII
    cii_overall = np.mean(cii_individual)

    return cii_overall, cii_individual