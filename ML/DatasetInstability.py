import numpy as np

from LR import LR_Model
from RF import RF_Classifier
from XGB import XGBoost_Model
from MLP import MLP_Model
from LSVC import LinearSVC_Model
from ML.init_database import init_dataset

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# init the dataset to train & test vars
[X_train, y_train, X_test, y_test] = init_dataset()

# set default random seed
random_seed = 42    

# kfold init
k_fold = KFold(n_splits=10, shuffle=True, random_state=random_seed)
    
for fold_id, (train_idx, test_idx) in enumerate(k_fold.split(X_train)):
    X_pool = X_train[train_idx]
    y_pool = y_train[train_idx]
    
    X_val = X_train[test_idx]
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
        RF_Classifier(X, y, fold_id, random_seed, b)
        XGBoost_Model(X, y, fold_id, random_seed, b)
        
        # fit the training data and transform the test data
        scaler = StandardScaler()
        X_boot_scaled = scaler.fit_transform(X_boot)
        X_val_transf = scaler.transform(X_val)
        X_test_transf = scaler.transform(X_test)
        X_scaled = [X_boot_scaled, X_val_transf, X_test_transf]
        
        # Train LR, MLP, and LinearSVC models
        LR_Model(X_scaled, y, fold_id, random_seed, b)
        MLP_Model(X_scaled, y, fold_id, random_seed, b)
        LinearSVC_Model(X_scaled, y, fold_id, random_seed, b)
        
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