import pandas as pd
import numpy as np
import os

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


from create_dir import create_nested_directory

class RuntimeMetrics:
    def __init__(
        self, 
        X_train, y_train,   # Training data (not split)
        X_test, y_test,     # Testing data
        validation_indices, train_indices, # indices
        instability_type,   # Dataset or Stochastic
        no_of_folds,        
        bootstrap_indices=None  # Optional
    ):
        # Official Training data (not split yet to validation and bootstrapped samples)
        self.X_train = X_train
        self.y_train = y_train
        
        # Official Testing  Data
        self.X_test = X_test
        self.y_test = y_test
        
        # Indices (for recreating the training and validation data)
        self.validation_indices = validation_indices
        self.train_indices = train_indices
        self.bootstrap_indices = bootstrap_indices
        
        # Consistent Data
        self.instability_type = instability_type
        self.no_of_folds = no_of_folds
        
        # Variables changed during metric calls
        self.csv_file_path = ''
        self.metric_result = ''
        self.metric_name = ''
        self.metric_directory = ''    
        
        
    # Recreates the validation data and bootstrapped training data to be used for the metrics
    def recreate_data_by_val_index(
        self, 
        train_idx,
        val_idx
    ):
        # We'll be saving the boot data as one whole array since this function gets data per fold
        X_boot_arr = []
        y_boot_arr = []
        
        # Making the training data separate from the validation data through val_idx and train_idx
        X_train_split = self.X_train.iloc[train_idx]
        y_train_split = self.y_train[train_idx]
        
        val_defendant_ids = self.X_train.iloc[val_idx]['defendant_id'].values # we get this
        X_val = self.X_train.iloc[val_idx]  # we get this
        y_val = self.y_train[val_idx]       # also this
        
        for boot_idx in self.bootstrap_indices:
            X_boot = X_train_split[boot_idx]
            X_boot_arr.append(X_boot)
            
            y_boot = y_train_split[boot_idx]
            y_boot_arr.append(y_boot)
        
        return [X_boot_arr, y_boot_arr, X_val, y_val, val_defendant_ids]
    
    # Reliability metric
    # Outside of the class since it's a separate experiment
    def calculate_mrip(
        self,
        X_target_ids,
        trained_model,
        model_name,
        test_name,
        fold_id,
        run,
        epsilon=0.1,
        delta=0.05
    ):  
        if model_name == 'RF' or model_name == 'XGB':
            scaler = StandardScaler()
            X_boot_scaled = scaler.fit_transform(X[0])
            X_val_transf = scaler.transform(X[1])
            X_test_transf = scaler.transform(X[2])
            X_scaled = [X_boot_scaled, X_val_transf, X_test_transf]
        
        mrip_values = []
        nn = NearestNeighbors(radius=delta)
        nn.fit(X_scaled[0]) # X train scaled
        
        distances, indices = nn.radius_neighbors(X_scaled[1 if test_name == 'Validation' else 2]) # X val scaled
        
        for i, defendant_id in enumerate(X_target_ids): # per fold, and every fold has N numbers of defendants
            neighbor_indices = indices[i]
            X_star = X[0].iloc[neighbor_indices]
            y_star = y[0].iloc[neighbor_indices]
            
            probabilities = trained_model.predict_proba(X_star)[:, 1]
            errors = np.abs(y_star - probabilities)
            mrip_value = np.mean(errors <= epsilon)
            mrip_values.append(mrip_value)        

        pd.DataFrame({
            'defendant_id': X_target_ids,
            f'MRIP_{run}': mrip_values
        })
    
    
                            
    def shap_analysis(self):
        print('hello wurl')
                
    # ===== HELPER FUNCTIONS =====
    def init_csv(
        self,
        column_names,
        row_count,
        model_name,
        test_name,
        fold_id=None,
        first_column_data=None,
    ):
        file_directory = f'metrics/{self.instability_type}/{self.metric_name}'
        create_nested_directory(file_directory)
        
        if first_column_data is None:
            df = pd.DataFrame(
                index=[f"{i}" for i in range(1, row_count + 1)],
                columns=column_names
            )
        else:
            df = pd.DataFrame(
                {column_names[0]: first_column_data},
                columns=column_names
            )
        fold_name = f'-Fold_{fold_id}' if fold_id != None else ''
        file_name = f'{self.metric_name}-{model_name}-{test_name}{fold_name}.csv'
        csv_file_path = f'{file_directory}/{file_name}'
        df.to_csv(csv_file_path, index=False)
        
        return csv_file_path
    
    def save_to_csv(
        self, 
        metric_result,
        metric_column_name,
        row=None
    ):
        if os.path.isfile(self.csv_file_path):
            results_csv = pd.read_csv(self.csv_file_path)
            
            if row is None:
                results_csv[metric_column_name] = metric_result
            else:
                results_csv.loc[row, metric_column_name] = metric_result
        results_csv.to_csv(self.csv_file_path, index=False)
  
    
    

    
    
    

   