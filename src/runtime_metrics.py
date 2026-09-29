import pandas as pd
import numpy as np
import os
import shap

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Reliability Metric
# (GOOD)
def MRIP(
    X_train,
    y_train,
    X_val_test,
    X_val_test_ids,
    trained_model, 
    model_name,
    nth_run,
    csv_file_path,
    epsilon=0.1,
    delta=0.5
):
    print(f'MRIP for model {model_name}')
    # Scale only for neighbor-distance calculation
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_test_scaled = scaler.transform(
        X_val_test
    )
    
    nn = NearestNeighbors(radius=delta)
    nn.fit(X_train_scaled)

    nearest_distances, indices = nn.radius_neighbors(X_val_test_scaled)
    
    mrip_values = []

    no_neighbor_count = 0
    for i, defendant_id in enumerate(X_val_test_ids):
        neighbor_indices = indices[i]

        if len(neighbor_indices) == 0:
            no_neighbor_count += 1  
            mrip_value = np.nan

        else:
            # Use the representation the model was trained on
            if model_name in ['LR', 'LSVC', 'MLP']:
                X_star = X_train_scaled[neighbor_indices]
            else:  # RF, XGB
                X_star = X_train.iloc[neighbor_indices]
            y_star = y_train[neighbor_indices]

            # All models now provide predict_proba()
            probabilities = trained_model.predict_proba(X_star)[:, 1]

            errors = np.abs(y_star - probabilities)

            mrip_value = np.mean(errors <= epsilon)

        mrip_values.append(mrip_value)

    print(
        f'No-neighbor defendants: '
        f'{no_neighbor_count}/{len(X_val_test_ids)}'
    )
    
    # Create or update CSV
    if os.path.isfile(csv_file_path):
        df = pd.read_csv(csv_file_path)
        df[f'MRIP_{nth_run}'] = mrip_values

    else:
        df = pd.DataFrame({
            'Defendant_ID': X_val_test_ids,
            f'MRIP_{nth_run}': mrip_values
        })

    df.to_csv(csv_file_path, index=False)
                       
def get_shap_values(
    X_train,
    X_val_test,
    feature_names,
    trained_model,
    model_name,
    nth_run,
    defendant_ids,
    csv_file_path
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
        f'Bootstrap_{nth_run}': shap_values.flatten()
    })

    print(f'getting shap values - csv_file_path: {csv_file_path}')
    df.to_csv(csv_file_path, index=False)