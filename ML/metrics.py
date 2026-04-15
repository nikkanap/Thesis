import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, auc, brier_score_loss
from statsmodels.nonparametric.smoothers_lowess import lowess
from top_k_jaccard import topk_jaccard
from CII import compute_cii

# For visualization purposes
def make_scatter():    
    models = ['LR', 'RF', 'XGB', 'MLP']
    for model in models:
        data = pd.read_csv(f'predictions/{model}_Predictions.csv')
        x_values = data['original_pred'].to_numpy()
    
        for j in range(1, 10):
            y_values = data[f'bootstrapped_{j}'].to_numpy()
            
            plt.scatter(x_values, y_values, color='blue', label=f'BS_f{j}')
            plt.xlabel('Original Predictions')
            plt.ylabel('BS Predictions')
            plt.title(f'Original vs Boostrapped {j} Predictions')
            plt.legend()
            
            directory = f'images/scatter/{model}/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            plt.savefig(f'{directory}{model}_BS_{j}.png')
            plt.close()
            
def generate_accuracy(y_test, y_pred):
    # GETTING THE ACCURACY:
    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy: {round(accuracy,2)}")

def generate_cm(y_test, y_pred):
    # GETTING THE CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_pred)
    cm
    
def generate_precision(y_test, y_pred):
    # GETTING THE PRECISION OF THE MODEL
    precision = precision_score(y_test, y_pred)
    print(f"Precision: {round(precision,2)}")

def generate_recall(y_test, y_pred):
    # HOW MANY ACTUAL POSITIVES WERE IDENTIFIED BY THE MODEL 
    recall = recall_score(y_test, y_pred)
    print(f"Recall: {round(recall,2)}")

def generate_f1(y_test, y_pred):
    f1 = f1_score(y_test, y_pred)
    print(f"f1 score: {round(f1,2)}")

# PART OF METRICS (Review tho)
def compute_pr_auc(y_test, y_prob):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    return auc(recall, precision)

def pr_auc_sd(y_test):
    models = ['LR', 'RF', 'XGB', 'MLP']
    
    print("SD of PR-AUC")
    for model in models:
        df = pd.read_csv(f'predictions/{model}_Predictions.csv')
        pr_aucs = []

        for col in df.columns:
            y_pred_proba = df[col].values
            pr_auc = compute_pr_auc(y_test, y_pred_proba)
            pr_aucs.append(pr_auc)
        
        pr_auc_sd = np.std(pr_aucs)
        print(f"{model}\t{pr_auc_sd:.4f}")   
        #return np.std(pr_aucs), pr_aucs
    
    """
    directory = 'images/ROC_curve/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    plt.figure()
    plt.plot(recall, precision, label=f"Precision-Recall Curve (AUC = {auc_score:.2f})")
    #plt.plot([0, 1], [0, 1], linestyle="--") 
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR-AUC Curve")
    plt.legend()
    
    directory = f'images/PR-AUC_curve/{model}/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(f"{directory}{model}_PR-AUC_{i}.png")
    plt.close()
    """

def brier_score_sd(y_test):
    models = ['LR', 'RF', 'XGB', 'MLP']
    
    print("Brier Score (Mean ± SD)")
    for model in models:
        df = pd.read_csv(f'predictions/{model}_Predictions.csv')
        brier_scores = []

        for col in df.columns:
            y_pred_proba = df[col].values
            # Compute Brier score for this run
            score = brier_score_loss(y_test, y_pred_proba)
            brier_scores.append(score)
        
        # Compute mean and SD of Brier scores across runs
        mean_brier = np.mean(brier_scores)
        sd_brier = np.std(brier_scores)
        print(f"{model}\tMean: {mean_brier:.4f}, SD: {sd_brier:.4f}")

# PART OF METRICS
def percentile():
    models = ['LR', 'RF', 'XGB', 'MLP']
    
    for model in models:
        df = pd.read_csv(f'predictions/{model}_Predictions.csv')
        original_pred = df["original_pred"].to_numpy()
        
        sorted_idx = np.argsort(original_pred)

        bootstrap_pred = df.drop(columns='original_pred').to_numpy()
        
        x_sorted = original_pred[sorted_idx]
        lower = np.percentile(bootstrap_pred, 2.5, axis=1)
        lower_sorted = lower[sorted_idx]
        upper = np.percentile(bootstrap_pred, 97.5, axis=1)
        upper_sorted = upper[sorted_idx]
        median = np.median(bootstrap_pred, axis=1)
        median_sorted = median[sorted_idx]
        
        # add lowess to make it smooth otherwise it'll look sharp and jagged
        lower_smooth = lowess(lower_sorted, x_sorted, frac=0.1)[:,1]
        upper_smooth = lowess(upper_sorted, x_sorted, frac=0.1)[:,1]
        median_smooth = lowess(median_sorted, x_sorted, frac=0.1)[:,1]
        
        plt.figure(figsize=(10, 6)) #10 by 6 inches
        plt.fill_between(x_sorted, lower_smooth, upper_smooth, color='lightgreen', alpha=0.4, label='95% stability interval')
        plt.plot(x_sorted, median_smooth, color="blue", label="Median prediction")
        plt.xlabel("Individual")
        plt.ylabel("Predicted Probability")
        plt.title("95% Stability Interval per Individual")
        plt.legend()
        
        directory = 'images/percentile/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f'{directory}{model}_percentile.png')
        plt.close()
        
#PART OF METRICS
def MAPE(): # check notebook lm to get the steps for MAPE values
    models = ['LR', 'RF', 'XGB', 'MLP']
    
    for model in models:
        df = pd.read_csv(f'predictions/{model}_Predictions.csv')
        predictions = df.iloc[:, 1:].values
        
        mean_preds = np.mean(predictions, axis=1)
        mape_indiv = np.mean(np.abs(predictions - mean_preds[:, np.newaxis]), axis=1)
        mape_overall = np.mean(mape_indiv)
        print(f"[{model}] Individual MAPE values: ")
        print(mape_indiv)
        
        print(f"[{model}] Overall MAPE: ")
        print(mape_overall)
        
# PART OF METRICS
def top_k_jaccard():
    models = ["LR", "MLP", "RF", "XGB"]

    for model in models:
        file = f"predictions/{model}_Predictions.csv"
        avg_jaccard, scores = topk_jaccard(file)

        print(f"{model} Average Top-10% Jaccard: {avg_jaccard:.4f}")
        
# PART OF METRICS
def cii():
    models = ["LR", "MLP", "RF", "XGB"]

    for model in models:
        file = f"predictions/{model}_Predictions.csv"
        cii_avg, cii_indiv = compute_cii(file)

        print(f"{model} CII: {cii_avkg:.4f}")
        
        plt.hist(cii_indiv, bins=20)
        plt.xlabel("CII per Individual")
        plt.ylabel("Count")
        plt.title(f"Classification Instability Distribution ({model})")
        directory = 'images/cii_distribution_plot/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f'{directory}{model}_distribution_plot.png')
        plt.close()
        
        