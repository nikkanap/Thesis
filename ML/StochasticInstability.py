from LR import LR_Model
from RF import RF_Classifier
from XGB import XGBoost_Model
from MLP import MLP_Model
from LSVC import LinearSVC_Model
from ML.init_database import init_dataset

import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# init the dataset to train & test vars
[X_train, y_train, X_test, y_test] = init_dataset()

# kfold init
k_fold = KFold(n_splits=10, shuffle=True, random_state=42)
    
for fold_id, (train_idx, test_idx) in enumerate(k_fold.split(X_train)):
    X_pool = X_train[train_idx]
    y_pool = y_train[train_idx]
    
    X_val = X_train[test_idx]
    y_val = y_train[test_idx]

    # random seed from 0 to 9
    for random_seed in range(10): 
        print(f"RANDOM_SEED #{random_seed}" )
        X = [X_train, X_val, X_test] 
        y = [y_train, y_val, y_test]
        
        RF_Classifier(X, y, fold_id, random_seed)
        
        # convert data for xgboost
        xgb_train = xgb.DMatrix(X[0], y[0], enable_categorical=False)
        xgb_val = xgb.DMatrix(X[1], y[1], enable_categorical=False)
        xgb_test = xgb.DMatrix(X[2], y[2], enable_categorical=False)
        X_xgb = [xgb_train, xgb_val, xgb_test]
        XGBoost_Model(X_xgb, y, fold_id, random_seed)
        
        # fit training data and transform the test data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_transf = scaler.transform(X_val)
        X_test_transf = scaler.transform(X_test)
        X_scaled = [X_train_scaled, X_val_transf, X_test_transf]
        
        LR_Model(X_scaled, y, fold_id, random_seed)
        MLP_Model(X_scaled, y, fold_id, random_seed)
        LinearSVC_Model(X_scaled, y, fold_id, random_seed)
        
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