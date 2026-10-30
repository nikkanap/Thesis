import pandas as pd
import numpy as np
import shap

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Reliability Metric
def MRIP(
    X_train,
    X_reference,
    y_reference,
    X_val_test,
    X_val_test_ids,
    trained_model, 
    model_name,
    fold_id,
    nth_run,
    epsilon=0.1,
    delta=0.5
):
    # Use the fitted model's training representation for both distances and
    # predictions. The reference examples must be held out from model fitting.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_reference_scaled = scaler.transform(X_reference)
    X_val_test_scaled = scaler.transform(X_val_test)

    nn = NearestNeighbors(radius=delta)
    nn.fit(X_reference_scaled)

    _, indices = nn.radius_neighbors(X_val_test_scaled)

    records = []

    for i, defendant_id in enumerate(X_val_test_ids):
        neighbor_indices = indices[i]

        if len(neighbor_indices) == 0:
            mrip_value = np.nan
        else:
            if model_name in ['LR', 'LSVC', 'MLP']:
                X_star = X_reference_scaled[neighbor_indices]
            else:  # RF, XGB
                X_star = X_reference.iloc[neighbor_indices]

            y_star = np.asarray(y_reference)[neighbor_indices]
            probabilities = trained_model.predict_proba(X_star)[:, 1]
            errors = np.abs(y_star - probabilities)
            mrip_value = np.mean(errors <= epsilon)

        records.append({
            'Defendant_ID': defendant_id,
            'Fold': fold_id,
            'Run': nth_run,
            'MRIP': mrip_value,
            'Neighbor_Count': len(neighbor_indices),
            'Epsilon': epsilon,
            'Delta': delta,
        })
    return records


def save_mrip_results(records, csv_file_path, run_label='Run'):
    """Save one row per evaluated defendant and fold, with runs as columns."""
    if not records:
        pd.DataFrame(columns=[
            'Defendant_ID', 'Fold', 'Mean MRIP'
        ]).to_csv(csv_file_path, index=False)
        return

    df = pd.DataFrame(records)
    run_ids = sorted(df['Run'].dropna().unique())
    run_columns = [f'{run_label} {run_id}' for run_id in run_ids]
    df['Run column'] = df['Run'].map(
        {run_id: column for run_id, column in zip(run_ids, run_columns)}
    )

    # Keep fold in the row key so test-set summaries are not averaged across
    # folds. Validation defendants are already selected separately per fold.
    summary = df.pivot(
        index=['Fold', 'Defendant_ID'],
        columns='Run column',
        values='MRIP',
    ).reset_index()
    summary.columns.name = None
    summary['Mean MRIP'] = summary[run_columns].mean(axis=1)
    summary = summary[[
        'Defendant_ID', 'Fold', *run_columns, 'Mean MRIP'
    ]]
    summary.to_csv(csv_file_path, index=False)
                       
def get_shap_values(
    X_train,
    X_val_test,
    feature_names,
    trained_model,
    model_name,
    nth_run,
    defendant_ids,
    run_label='Run'
):
    if model_name in ['RF', 'XGB']:
        explainer = shap.TreeExplainer(trained_model)
        shap_values = explainer.shap_values(X_val_test)

    elif model_name in ['LR', 'LSVC']:        
        if model_name == 'LSVC':
            explainer = shap.Explainer(
                trained_model.predict_proba,
                X_train
            )
            shap_values = explainer(X_val_test).values
        else:
            explainer = shap.LinearExplainer(
                trained_model,
                X_train
            )
            shap_values = explainer.shap_values(X_val_test)

    elif model_name == 'MLP':
        # Smaller background dataset
        background = shap.sample(
            X_train,
            min(50, len(X_train)),
            random_state=42
        )

        explainer = shap.KernelExplainer(
            trained_model.predict_proba,
            background
        )

        # Explicit computational budget
        shap_values = explainer.shap_values(
            X_val_test,
            nsamples=100
        )

    # Handle binary classification SHAP output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    df = pd.DataFrame({
        'defendant_id': np.repeat(defendant_ids, len(feature_names)),
        'feature': np.tile(feature_names, len(defendant_ids)),
        f'{run_label}_{nth_run}': shap_values.flatten()
    })
    return df


def save_shap_values(shap_run_frames, csv_file_path):
    """Save SHAP values from multiple runs in aligned run-specific columns."""
    if not shap_run_frames:
        pd.DataFrame(columns=['defendant_id', 'feature']).to_csv(
            csv_file_path,
            index=False
        )
        return

    indexed_run_frames = [
        run_frame.set_index(['defendant_id', 'feature'])
        for run_frame in shap_run_frames
    ]
    base_index = indexed_run_frames[0].index
    if not base_index.is_unique:
        raise ValueError('Each SHAP run must have one value per defendant and feature.')

    for run_frame in indexed_run_frames:
        if len(run_frame.columns) != 1 or not run_frame.index.equals(base_index):
            raise ValueError('SHAP runs must contain the same defendant and feature rows.')

    run_columns = [run_frame.columns[0] for run_frame in indexed_run_frames]
    run_values = np.column_stack([
        run_frame.iloc[:, 0].to_numpy()
        for run_frame in indexed_run_frames
    ])
    combined_columns = {
        'defendant_id': base_index.get_level_values('defendant_id'),
        'feature': base_index.get_level_values('feature'),
        **{
            run_column: run_values[:, column_index]
            for column_index, run_column in enumerate(run_columns)
        },
    }
    combined = pd.DataFrame(combined_columns)
    combined.to_csv(csv_file_path, index=False)
