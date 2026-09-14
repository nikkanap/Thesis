import pandas as pd
import numpy as np
import os
import shap

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from create_dir import create_nested_directory

# Reliability Metric
# (GOOD)
def MRIP(
    X_train, y_train, X_val_test,
    instability_type,
    defendant_ids,
    trained_model, model_name,
    test_name, fold_id, nth_run,
    epsilon=0.1,
    delta=0.05
):
    mrip_dir = f'metrics/{instability_type}/MRIP'
    create_nested_directory(mrip_dir)

    file_name = f'MRIP_{model_name}_{test_name}_Fold_{fold_id}.csv'
    csv_file_path = f'{mrip_dir}/{file_name}'

    # Scale only for neighbor-distance calculation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_test_scaled = scaler.transform(X_val_test)

    # Find training neighbors within delta
    nn = NearestNeighbors(radius=delta)
    nn.fit(X_train_scaled)

    distances, indices = nn.radius_neighbors(X_val_test_scaled)

    mrip_values = []

    for i, defendant_id in enumerate(defendant_ids):
        neighbor_indices = indices[i]

        if len(neighbor_indices) == 0:
            mrip_value = np.nan
        else:
            X_star = X_train.iloc[neighbor_indices]
            y_star = y_train.iloc[neighbor_indices]

            if model_name == 'XGB':
                probabilities = trained_model.predict(X_star)
            else:
                probabilities = trained_model.predict_proba(X_star)[:, 1]

            errors = np.abs(y_star - probabilities)
            mrip_value = np.mean(errors <= epsilon)

        mrip_values.append(mrip_value)

    # Create or update CSV
    if os.path.isfile(csv_file_path):
        df = pd.read_csv(csv_file_path)
    else:
        df = pd.DataFrame({
            'Defendant_ID': defendant_ids
        })

    df[f'MRIP_{nth_run}'] = mrip_values
    df.to_csv(csv_file_path, index=False) 
                       
def get_shap_values(
    X_train,
    X_val_test,
    trained_model,
    model_name,
    test_name,
    fold_id,
    nth_run,
    instability_type,
    defendant_ids
):
    shap_dir = f'shap_values/{instability_type}'
    create_nested_directory(shap_dir)
    
    file_name = f'Shap_Values_{model_name}_{test_name}_Fold_{fold_id}.csv'
    csv_file_path = f'{shap_dir}/{file_name}'
    if model_name in ['RF', 'XGB']:
        explainer = shap.TreeExplainer(trained_model)
        shap_values = explainer.shap_values(X_val_test)
    elif model_name in ['LR', 'LSVM']:
        explainer = shap.LinearExplainer(trained_model, X_train)
        shap_values = explainer.shap_values(X_val_test) 
    elif model_name == 'MLP':
        background = shap.sample(X_train, 100)
        explainer = shap.KernelExplainer(
            trained_model.predict_proba,
            background
        )
        shap_values = explainer.shap_values(X_val_test)

    if nth_run == 1:
        df = pd.DataFrame({ 
            'defendant_id' : defendant_ids,
            f'fold_{fold_id}': shap_values
        })
    else:
        df = pd.read_csv(csv_file_path)
        df[f'fold_{fold_id}'] = shap_values
    df.to_csv(csv_file_path, index=False)

    
    
    

   