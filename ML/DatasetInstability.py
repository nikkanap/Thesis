from datasets import load_dataset
import pandas as pd
import numpy as np

from LR import LR_Model
from RF import RF_Classifier
from XGB import XGBoost_Model
from MLP import MLP_Model
from ML.LSVC import LinearSVC_Model
from metrics import make_scatter, percentile, MAPE, top_k_jaccard, cii, pr_auc_sd, brier_score_sd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# loading the dataset
dataset = load_dataset("imodels/compas-recidivism")
training_df = pd.DataFrame(dataset['train'])  # get the training data
testing_df = pd.DataFrame(dataset['test'])    # get the testing data

# adding permanent identifier ids for metrics
training_df["defendant_id"] = range(len(training_df))
testing_df["defendant_id"] = range(len(testing_df))

# kfold init
k_fold = KFold(n_splits=10, shuffle=True, random_state=42)
# separate the training data into X_train and y_train
X_train = training_df.drop(columns=['is_recid'])
y_train = training_df['is_recid'].to_numpy()

# same goes for the test data
X_test = testing_df.drop(columns=['is_recid'])
y_test = testing_df['is_recid'].to_numpy()
    
for fold_id, (train_idx, test_idx) in enumerate(k_fold.split(X_train)):
    X_pool = X_train[train_idx]
    y_pool = y_train[train_idx]
    
    X_val = X_train[test_idx]
    y_val = y_train[test_idx]

    # bootstrapping for 21 rows la anay
    for b in range(10): 
        print(f"ITERATION {b}" )
        
            # make bootstrapped sample of training dataset
        n_samples = len(X_pool)
        boot_idx = np.random.choice(
            n_samples, 
            size=n_samples, 
            replace=True
        )
        X_boot = X_pool.iloc[boot_idx]
        y_boot = y_pool[boot_idx]
        
        data = [
            X_boot, y_boot,
            X_val, y_val,
            X_test, y_test,
            b
        ]
        
        RF_Classifier(data)
        XGBoost_Model(data)
        
        # fit the training data and transform the test data
        scaler = StandardScaler()
        X_boot_scaled = scaler.fit_transform(X_boot)
        X_val_transf = scaler.transform(X_val)
        X_test_transf = scaler.transform(X_test)

        data_transf = [
            X_boot_scaled, y_boot,
            X_val_transf, y_val,
            X_test_transf, y_test,
            fold_id, b
        ]
        
        LR_Model(data_transf)
        MLP_Model(data_transf)
        LinearSVC_Model(data_transf)
        
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