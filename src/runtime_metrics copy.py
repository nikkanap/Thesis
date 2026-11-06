import pandas as pd
import numpy as np
import shap

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Constant variables
MRIP_EPSILON = 0.15
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
    X_ref_scaled = scaler.fit_transform(X_ref)
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


# 🗹 Saves the MRIP results in csv files
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

# 🗹 Gets the raw shap values per run
def get_shap_values(
    X_ref,
    chosen_X_eval,
    chosen_eval_ids,
    feature_names,
    model_name,
    trained_model,
    nth_run,
    run_label='Run'
):
    if model_name in ['RF', 'XGB']:
        X_ref_sample = shap.sample(
            X_ref,
            min(100, len(X_ref)),
            random_state=42,
        )
        explainer = shap.TreeExplainer(
            trained_model,
            data=X_ref_sample,
            feature_perturbation="interventional",
            model_output="probability"
        )
        shap_values = explainer.shap_values(chosen_X_eval)

    elif model_name in ['LR', 'LSVC']:        
        if model_name == 'LSVC':
            explainer = shap.Explainer(
                trained_model.predict_proba,
                X_ref,
                seed=42
            )
            shap_values = explainer(chosen_X_eval).values
        else:
            # LR
            explainer = shap.LinearExplainer(
                trained_model,
                X_ref
            )
            shap_values = explainer(chosen_X_eval).values

    elif model_name == 'MLP':
        # Smaller background dataset
        X_ref_sample = shap.sample(
            X_ref,
            min(100, len(X_ref)),
            random_state=42
        )

        explainer = shap.KernelExplainer(
            trained_model.predict_proba,
            X_ref_sample
        )

        shap_values = explainer.shap_values(
            chosen_X_eval,
            nsamples=100
        )

    # Handle binary classification SHAP output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    return {
        'Defendant_ID': np.repeat(chosen_eval_ids, len(feature_names)),
        'Feature': np.tile(feature_names, len(chosen_eval_ids)),
        f'{run_label}_{nth_run}': shap_values.flatten()
    }

# 🗹 Saves the raw shap values in csv files
def save_shap_values(
    shap_records, 
    csv_file_path
):
    # Convert first dict into a df
    shap_df = pd.DataFrame(shap_records[0])

    # Add the remaining runs' SHAP values
    for record in shap_records[1:]:
        run = pd.DataFrame(record)

        # Check that defendants and features match
        if not run[['Defendant_ID', 'Feature']].equals(
            shap_df[['Defendant_ID', 'Feature']]
        ):
            raise ValueError('SHAP runs must contain the same defendants and features.')

        # Get the SHAP column name
        run_column = run.columns[2]

        # Add the SHAP values to the combined DataFrame
        shap_df[run_column] = run[run_column]

    # Save to CSV
    shap_df.to_csv(csv_file_path, index=False)
