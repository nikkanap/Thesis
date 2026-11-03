import pandas as pd
import numpy as np
import shap

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Constant variables
MRIP_EPSILON = 0.1
MRIP_DELTA = 0.5

# ==================================================
# MODEL RELIABILITY FOR INDIVIDUAL PREDICTION (MRIP)
# ==================================================

def MRIP(
    X_ref,
    y_ref,
    chosen_X_eval, 
    chosen_eval_ids,
    model_name, 
    trained_model, 
    fold_id, 
    nth_run
):
    # Scale the X_ref and chosen_X_eval
    scaler = StandardScaler()
    X_ref_scaled = scaler.transform(X_ref)
    chosen_X_eval_scaled = scaler.transform(chosen_X_eval)

    # Fit the nearest neighbors on the actual training data
    nn = NearestNeighbors(radius=MRIP_DELTA)
    nn.fit(X_ref_scaled)

    _, indices = nn.radius_neighbors(chosen_X_eval_scaled)

    mrip_list = []
    for i, defendant_id in enumerate(chosen_eval_ids):
        neighbor_indices = indices[i]

        if len(neighbor_indices) == 0:
            mrip_value = np.nan
        else:
            X_star = (
                X_ref_scaled[neighbor_indices]
                if model_name in ['LR', 'LSVC', 'MLP']
                else X_ref.iloc[neighbor_indices]
            )
            y_star = np.asarray(y_ref)[neighbor_indices]
            probabilities = trained_model.predict_proba(X_star)[:, 1]
            
            errors = np.abs(y_star - probabilities)
            mrip_value = np.mean(errors <= MRIP_EPSILON)

        mrip_list.append({
            'Defendant_ID': defendant_id,
            'Fold': fold_id,
            'Run': nth_run,
            'MRIP': mrip_value,
            'Neighbor_Count': len(neighbor_indices),
        })
    return mrip_list


# Saves the MRIP results in csv files
def save_mrip_results(
    records, 
    csv_file_path, 
    run_label='Run'
):
    df = pd.DataFrame(records)

    # Get run IDs and their corresponding column names
    run_ids = sorted(df['Run'].unique())
    run_columns = [f'{run_label} {run_id}' for run_id in run_ids]

    # Convert rows into columns for each run
    mrip_summary = df.pivot(
        index=['Fold', 'Defendant_ID'],
        columns='Run',
        values='MRIP'
    ).reset_index()

    # Rename run ID columns and clean the column's axis name (runs)
    mrip_summary = mrip_summary.rename(
        columns=dict(zip(run_ids, run_columns))
    )
    mrip_summary.columns.name = None

    # Calculate mean MRIP across runs
    mrip_summary['Mean MRIP'] = mrip_summary[run_columns].mean(axis=1)
    mrip_summary.to_csv(csv_file_path, index=False)
                  
# ====================================
# SHAPLEY ADDITIVE EXPLANATIONS (SHAP)
# ====================================
                       
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


# Save SHAP values from multiple runs in aligned run-specific columns.
def save_shap_values(shap_run_frames, csv_file_path):
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
