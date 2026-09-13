import pandas as pd
import numpy as np
import os

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from create_dir import create_nested_directory

# Reliability Metric
# (GOOD)
def MRIP(
    X_train,
    y_train,
    X_val_test,
    instability_type,
    defendant_ids,
    trained_model,
    model_name,
    test_name,
    fold_id,
    nth_run,
    epsilon=0.1,
    delta=0.05
):
    file_directory = f'metrics/{instability_type}/MRIP'
    create_nested_directory(file_directory)

    file_name = f'MRIP_{model_name}_{test_name}_Fold_{fold_id}.csv'
    csv_file_path = f'{file_directory}/{file_name}'

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
                       
def shap_analysis():
    print('hello wurl') 
    
    

    
    
    

   