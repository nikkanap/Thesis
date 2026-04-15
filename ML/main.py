from datasets import load_dataset
import pandas as pd
import numpy as np

from LR import LR_Model
from RF import RF_Classifier
from XGB import XGBoost_Model
from MLP import MLP_Model
from metrics import make_scatter, percentile, MAPE, top_k_jaccard, cii, pr_auc_sd, brier_score_sd

from sklearn.preprocessing import StandardScaler

# loading the dataset
dataset = load_dataset("imodels/compas-recidivism")
training_df = pd.DataFrame(dataset['train'])  # get the training data
testing_df = pd.DataFrame(dataset['test'])    # get the testing data

y_test = testing_df['is_recid'].to_numpy()
"""
for i in range(21): 
    # separate the training data into X_train and y_train
    print(f"ITERATION {i}" )
    X_train = training_df.drop(columns=['is_recid'])
    y_train = training_df['is_recid'].to_numpy()
    
    # same goes for the test data
    X_test = testing_df.drop(columns=['is_recid'])
    y_test = testing_df['is_recid'].to_numpy()
    
    if i > 0:
        # make bootstrapped sample of training dataset
        n_samples = len(X_train)
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_train = X_train.iloc[bootstrap_indices]
        y_train = y_train[bootstrap_indices]

    # hulabulibi fit the training data and transform the test data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    LR_Model(X_train, y_train, X_test, y_test, i)
    RF_Classifier(X_train, y_train, X_test, y_test, i)
    XGBoost_Model(X_train, y_train, X_test, y_test, i)
    MLP_Model(X_train, y_train, X_test, y_test, i)
    
"""

# displaying as a scatter plot
make_scatter()
percentile()
MAPE()
top_k_jaccard()
cii()
pr_auc_sd(y_test)
brier_score_sd(y_test)