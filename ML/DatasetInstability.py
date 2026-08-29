import numpy as np

from LR import LR_Model
from RF import RF_Classifier
from XGB import XGBoost_Model
from MLP import MLP_Model
from LSVC import LinearSVC_Model
from init_database import init_dataset
from create_dir import create_nested_directory

import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# init the dataset to train & test vars
[X_train, y_train, X_test, y_test] = init_dataset()
create_nested_directory('predictions/Dataset')

# set default random seed
random_seed = 42    

# kfold init
k_fold = KFold(n_splits=10, shuffle=True, random_state=random_seed)
    
for fold_id, (train_idx, test_idx) in enumerate(k_fold.split(X_train)):
    X_pool = X_train.iloc[train_idx]
    y_pool = y_train[train_idx]
    
    X_val_ids = X_train.iloc[test_idx]['defendant_id'].values
    X_val = X_train.iloc[test_idx]
    y_val = y_train[test_idx]

    # bootstrapping for 10 rows la anay
    for b in range(10): 
        print(f"BOOTSTRAP #{b}" )
        
        # make bootstrapped sample of training dataset
        n_samples = len(X_pool)
        boot_idx = np.random.choice(
            n_samples, 
            size=n_samples, 
            replace=True
        )
        X_boot = X_pool.iloc[boot_idx]
        y_boot = y_pool[boot_idx]
        
        X = [X_boot, X_val, X_test] 
        y = [y_boot, y_val, y_test]
        
        # Train RF and XGBoost models
        RF_Classifier(X, y, X_val_ids, fold_id, random_seed, b)# convert data for xgboost
        
        xgb_train = xgb.DMatrix(X[0], y[0], enable_categorical=False)
        xgb_val = xgb.DMatrix(X[1], y[1], enable_categorical=False)
        xgb_test = xgb.DMatrix(X[2], y[2], enable_categorical=False)
        X_xgb = [xgb_train, xgb_val, xgb_test]
        XGBoost_Model(X_xgb, y, X_val_ids, fold_id, random_seed, b)
        
        # fit the training data and transform the test data
        scaler = StandardScaler()
        X_boot_scaled = scaler.fit_transform(X_boot)
        X_val_transf = scaler.transform(X_val)
        X_test_transf = scaler.transform(X_test)
        X_scaled = [X_boot_scaled, X_val_transf, X_test_transf]
        
        # Train LR, MLP, and LinearSVC models
        LR_Model(X_scaled, y, X_val_ids, fold_id, random_seed, b)
        MLP_Model(X_scaled, y, X_val_ids, fold_id, random_seed, b)
        LinearSVC_Model(X_scaled, y, X_val_ids, fold_id, random_seed, b)
    
"""
# displaying as a scatter plot
make_scatter()
percentile()
MAPE()
top_k_jaccard()
cii()
pr_auc_sd(y_test)
brier_score_sd(y_test)
"""