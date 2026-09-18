import numpy as np
import os

from models.LR import LR_Model
from models.RF import RF_Classifier
from models.XGB import XGBoost_Model
from models.MLP import MLP_Model
from models.LSVC import LinearSVC_Model
from init_database import init_dataset
from create_dir import create_nested_directory

import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from post_metrics import PostMetrics

# set default random seed
random_seed = 42    
no_of_folds = 3
bootstraps = 3
instability_type = 'Dataset'

# kfold init
k_fold = KFold(
    n_splits=no_of_folds, 
    shuffle=True, 
    random_state=random_seed
)
    
# Arrays for metrics later (we need to save these)
val_defendant_ids_arr = []
X_val_arr = [] # 10 total validation tests
y_val_arr = [] # 10 total validation true values

# init the dataset to train & test vars
[X_train, y_train, training_ids, X_test, y_test, testing_ids] = init_dataset()

# NOTE: Since we only have one test for everything, we just scale & DMatrix the test data once
test_defendant_ids = testing_ids

# Create the directory for saving the predictions
predictions_dir = f'predictions/{instability_type}'
create_nested_directory(predictions_dir)

for fold_id, (train_idx, val_idx) in enumerate(k_fold.split(X_train), start=1):    
    # Making the training data separate from the validation data (fold)
    X_train_split = X_train.iloc[train_idx]
    y_train_split = y_train[train_idx]
    
    val_defendant_ids = training_ids[train_idx]
    val_defendant_ids_arr.append(val_defendant_ids)
    
    X_val = X_train.iloc[val_idx]
    X_val_arr.append(X_val)
    
    y_val = y_train[val_idx]
    y_val_arr.append(y_val)

    for b in range(1, bootstraps + 1): 
        model_data = {}
        
        print(f"BOOTSTRAP #{b}" )
        
        # Make bootstrapped sample of training dataset
        n_samples = len(X_train_split)
        boot_idx = np.random.choice(
            n_samples, 
            size=n_samples, 
            replace=True
        )
        
        # Create the bootstrapped training data
        X_boot = X_train_split.iloc[boot_idx]
        y_boot = y_train_split[boot_idx]
        
        # For model training
        model_data = {
            'X_train': X_boot,
            'y_train': y_boot,
            'testing_data': [
                {   
                    'name': 'Validation',
                    'X_train_raw': X_boot,
                    'y_train': y_boot,
                    'X_raw': X_val,
                    'X': X_val,
                    'y': y_val,
                    'ids': val_defendant_ids
                },
                {
                    'name': 'Test',
                    'X_train_raw': X_boot,
                    'y_train': y_boot,
                    'X_raw': X_test,
                    'X': X_test,
                    'y': y_test,
                    'ids': test_defendant_ids
                }
            ]
        }
        
        # Train RF
        RF_Classifier(
            predictions_dir, 
            model_data, 
            instability_type, 
            fold_id, random_seed, b
        )
        
        # Create DMatrix of X_boot and X_val
        xgb_boot = xgb.DMatrix(X_boot, y_boot, enable_categorical=False)
        xgb_val = xgb.DMatrix(X_val, y_val, enable_categorical=False)
        xgb_test = xgb.DMatrix(X_test, y_test, enable_categorical=False)
        
        model_data['X_train'] = xgb_boot
        model_data['testing_data'][0]['X'] = xgb_val
        model_data['testing_data'][1]['X'] = xgb_test
        XGBoost_Model(
            predictions_dir, 
            model_data,
            instability_type, 
            fold_id, random_seed, b
        )
        
        # fit X_boot and transform the X_val 
        scaler = StandardScaler()
        X_boot_scaled = scaler.fit_transform(X_boot)
        X_test_transf = scaler.transform(X_test)    
        X_val_transf = scaler.transform(X_val)
        
        model_data['X_train'] = X_boot_scaled
        model_data['testing_data'][0]['X'] = X_val_transf
        model_data['testing_data'][1]['X'] = X_test_transf
        
        # Train LR, MLP, and LinearSVC models
        LR_Model(
            predictions_dir, 
            model_data,
            instability_type, 
            fold_id, random_seed, b
        )
        MLP_Model(
            predictions_dir, 
            model_data, 
            instability_type, 
            fold_id, random_seed, b
        )
        LinearSVC_Model(
            predictions_dir, 
            model_data, 
            instability_type, 
            fold_id, random_seed, b
        )


pm = PostMetrics(
        X_train, y_train,   
        X_val_arr, y_val_arr, val_defendant_ids_arr,
        X_test, y_test, test_defendant_ids,
        no_of_folds,        
        predictions_dir,
        'Dataset'
    )

pm.roc_auc()
pm.brier_score()
pm.calibration_plot()
pm.ninety_five_stability_interval()
pm.mean_absolute_prediction_error()
pm.classification_instability_index()
pm.top_k_jaccard()
pm.demographic_false_positive_rate()
pm.shap_analysis()
pm.shap_analysis()

