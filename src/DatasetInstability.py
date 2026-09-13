import numpy as np

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

# set default random seed
random_seed = 42    
no_of_folds = 10
bootstraps = 100

# kfold init
k_fold = KFold(
    n_splits=no_of_folds, 
    shuffle=True, 
    random_state=random_seed
)
    
# Arrays for metrics later (we need to save these)
# 100 individual bootstrap indices


# 10 validation indices
validation_indices = []
train_indices = []
bootstrap_indices = []

# NOTE: No need to make test_indices since there's really only one official test X and y

# init the dataset to train & test vars
[X_train, y_train, X_test, y_test] = init_dataset()

# Create the directory for saving the predictions
predictions_dir = 'predictions/dataset_instability'
create_nested_directory(predictions_dir)

for fold_id, (train_idx, val_idx) in enumerate(k_fold.split(X_train)):
    validation_indices.append(val_idx) # Saving the validation indices
    train_indices.append(train_idx)
    
    # Making the training data separate from the validation data (fold)
    X_train_split = X_train.iloc[train_idx]
    y_train_split = y_train[train_idx]
    
    val_defendant_ids = X_train.iloc[val_idx]['defendant_id'].values
    X_val = X_train.iloc[val_idx]
    y_val = y_train[val_idx]

    # bootstrapping for 10 rows la anay
    for b in range(1, bootstraps + 1): 
        print(f"BOOTSTRAP #{b}" )
        
        # make bootstrapped sample of training dataset
        n_samples = len(X_train_split)
        boot_idx = np.random.choice(
            n_samples, 
            size=n_samples, 
            replace=True
        )
        bootstrap_indices.append(boot_idx) # Saving the bootstrap indices
        
        X_boot = X_train_split.iloc[boot_idx]
        y_boot = y_train_split[boot_idx]
        
        # Train RF and XGBoost models
        RF_Classifier(predictions_dir, X_boot, y_boot, X_val, y_val, X_test, y_test, fold_id, random_seed, b)# convert data for xgboost
        
        xgb_train = xgb.DMatrix(X_boot, y_boot, enable_categorical=False)
        xgb_val = xgb.DMatrix(X_val, y_val, enable_categorical=False)
        xgb_test = xgb.DMatrix(X_test, y_test, enable_categorical=False)
        XGBoost_Model(predictions_dir, xgb_train, y_boot, xgb_val, y_val, xgb_test, y_test, fold_id, random_seed, b)
        
        # fit the training data and transform the test data
        scaler = StandardScaler()
        X_boot_scaled = scaler.fit_transform(X_boot)
        X_val_transf = scaler.transform(X_val)
        X_test_transf = scaler.transform(X_test)
        
        # Train LR, MLP, and LinearSVC models
        LR_Model(predictions_dir, X_boot_scaled, y_boot, X_val_transf, y_val, X_test_transf, y_test, fold_id, random_seed, b)
        MLP_Model(predictions_dir, X_boot_scaled, y_boot, X_val_transf, y_val, X_test_transf, y_test, fold_id, random_seed, b)
        LinearSVC_Model(predictions_dir, X_boot_scaled, y_boot, X_val_transf, y_val, X_test_transf, y_test, fold_id, random_seed, b)