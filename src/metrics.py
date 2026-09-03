import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import roc_auc_score, roc_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve
from statsmodels.nonparametric.smoothers_lowess import lowess
from top_k_jaccard import top_k_jaccard_all

from create_dir import create_nested_directory

# list of models being tested
models = ['LR', 'RF', 'XGB', 'LinearSVC' 'MLP']

# ===== HELPER FUNCTIONS =====
# saves results from all k_folds per model in one csv file
def save_to_csv_all_folds(metric_result, metric_directory, metric_name, instability_type, model_name, test_name, fold_id):
    csv_file_path = f'metrics/{instability_type}/{metric_directory}/{metric_name}_{model_name}_{test_name}.csv'
    create_nested_directory(f'metrics/{instability_type}/{metric_name}')
    
    if os.path.isfile(csv_file_path):
        predictions_with_id_df = pd.read_csv(csv_file_path)
        predictions_with_id_df[f'fold_{fold_id}'] = metric_result
    else:
        predictions_with_id_df = pd.DataFrame({
            f'fold_{fold_id}': metric_result
        })
    predictions_with_id_df.to_csv(csv_file_path, index=False)

# saves results from all runs in a k_fold per model csv file
def save_to_csv_by_fold(metric_result, metric_directory, metric_name, instability_type, model_name, test_name, fold_id, defendant_ids=None):
    csv_file_path = f'metrics/{instability_type}/{metric_directory}/{metric_name}_{model_name}_{test_name}_{fold_id}.csv'
    create_nested_directory(f'metrics/{instability_type}/{metric_name}')
    
    if os.path.isfile(csv_file_path):
        predictions_with_id_df = pd.read_csv(csv_file_path)
        predictions_with_id_df[metric_name] = metric_result
    else:
        predictions_with_id_df = pd.DataFrame({
            **({f'defendant_id': defendant_ids} if defendant_ids != None else {}),
            metric_name: metric_result
        })
    predictions_with_id_df.to_csv(csv_file_path, index=False)

# saves individual results per run, also in a k_fold per model csv file
def save_to_csv_by_run(metric_result, metric_directory, metric_name, instability_type, model_name, test_name, fold_id, col, defendant_ids=None):
    csv_file_path = f'metrics/{instability_type}/{metric_directory}/{metric_name}_{model_name}_{test_name}_{fold_id}.csv'
    create_nested_directory(f'metrics/{instability_type}/{metric_name}')
    
    if os.path.isfile(csv_file_path):
        predictions_with_id_df = pd.read_csv(csv_file_path)
        predictions_with_id_df[col] = metric_result
    else:
        predictions_with_id_df = pd.DataFrame({
            **({f'defendant_id': defendant_ids} if defendant_ids != None else {}),
            col: metric_result
        })
    predictions_with_id_df.to_csv(csv_file_path, index=False)

# For visualization purposes
'''
def generate_scatter():    
    for model in models:
        data = pd.read_csv(f'metrics/{model}_Predictions.csv')
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
'''

# ===== PART OF METRICS ===== 
def roc_auc_and_std(instability_type, test_name, y):
    print('Computing ROC-AUC and STD...')
    
    for model in models:
        roc_aucs = []
        
        for fold_id in range(10):
            predictions_with_id_df = pd.read_csv(f'predictions/{instability_type}/{model}_Predictions_{test_name}_{fold_id}.csv')
            predictions = predictions_with_id_df.drop(columns=['defendant_id']).to_numpy()
            defendant_ids = predictions_with_id_df['defendant_id']
                            
            for col in predictions.columns:
                y_pred_proba = predictions[col].values
                roc_auc = roc_auc_score(y, y_pred_proba)
                save_to_csv_by_run(roc_auc_mean, 'roc_auc_mean', instability_type, model, test_name, fold_id, col, defendant_ids)
                roc_aucs.append(roc_auc)
        
        roc_auc_mean = np.mean(roc_aucs)
        save_to_csv_by_fold(roc_auc_mean, 'roc_auc_mean', instability_type, model, test_name, fold_id, defendant_ids)
        
        roc_auc_sd = np.std(roc_aucs)
        save_to_csv_by_fold(roc_auc_sd, 'roc_auc_std', instability_type, model, test_name, fold_id, defendant_ids)
    
# Performance metric
# Gets the brier score
def brier_score(y, instability_type, tests, folds):
    for i in range(2):
        print(f'[Generating {instability_type}-{tests[i]} Brier Scores]')
        
        for model in models:
            for fold_id in folds:
                predictions_with_id_df = pd.read_csv(
                    f'predictions/{instability_type}/'
                    f'{model}_Predictions_{tests[i]}_{fold_id}.csv'
                )
                predictions = predictions_with_id_df.drop(columns=['defendant_id']).to_numpy()
                defendant_ids = predictions_with_id_df['defendant_id']
                brier_scores = []

                for col in predictions.columns:
                    y_pred_proba = predictions[col].values
                    
                    # Compute Brier score for this run and append the scores by run (bootstrap or random seed)
                    score = brier_score_loss(y[i], y_pred_proba)
                    brier_scores.append(score)
                    save_to_csv_by_run(score, 'brier_score', instability_type, model, tests[i], fold_id, col, defendant_ids)
                
                # Compute mean and SD of Brier scores across runs
                mean_brier = np.mean(brier_scores)
                save_to_csv_all_folds(mean_brier, 'mean_brier_score', instability_type, model, tests[i], fold_id)
                
                sd_brier = np.std(brier_scores)
                save_to_csv_all_folds(sd_brier, 'sd_brier_score', instability_type, model, tests[i], fold_id)

# Instability metric
# Gets the 95% Instability Percentile
def ninety_five_instability_percentile(instability_type, tests, folds):
    # index for tests = ['validation', 'test']
    for i in range(2):
        print(f'[Generating {instability_type}-{tests[i]} 95% Instability Percentile]')
        
        for model in models:
            for fold_id in folds:
                predictions_with_id_df = pd.read_csv(
                    f'predictions/{instability_type}/'
                    f'{model}_Predictions_{tests[i]}_{fold_id}.csv'
                )
                predictions = predictions_with_id_df.drop(columns=['defendant_id']).to_numpy()
                
                stability_df = pd.DataFrame({
                    'defendant_id': predictions_with_id_df['defendant_id'],
                    'mean_prediction': np.mean(predictions, axis=1),
                    'lower_95': np.percentile(predictions, 2.5, axis=1),
                    'upper_95': np.percentile(predictions, 97.5, axis=1)
                })
                
                stability_df['si_width'] = ( 
                    stability_df['upper_95'] - stability_df['lower_95']
                )
                
                plot_df = stability_df.head(100)
                plt.figure(figsize=(12, 6))
                plt.errorbar(
                    plot_df['defendant_id'],
                    plot_df['mean_prediction'],
                    yerr=[
                        plot_df['mean_prediction'] - plot_df['lower_95'],
                        plot_df['upper_95'] - plot_df['mean_prediction']
                    ],
                    fmt='o',
                    capsize=3
                )
                plt.xlabel('Defendant ID')
                plt.ylabel('Predicted Probability')
                plt.title(
                    f'{instability_type} - {model} - Fold {fold_id} - 95% Stability Intervals of Predicted Recidivism Risk'
                )
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.show()
                plt.savefig(
                    f'metrics/{instability_type}/95_percentile/'
                    f'95_Percentile_{model}_{tests[i]}_{fold_id}.png'
                )
        
# Instability metric
# Mean Absolute Prediction Error
def mean_absolute_prediction_error(y, instability_type, tests, folds): # check notebook lm to get the steps for MAPE values
    for i in range(2):
        for model in models:
            for fold_id in folds:
                predictions_with_id_df = pd.read_csv(
                    f'predictions/{instability_type}/'
                    f'{model}_Predictions_{tests[i]}_{fold_id}.csv'
                )
                predictions = predictions_with_id_df.drop(columns=['defendant_id']).to_numpy()
                defendant_ids = predictions_with_id_df['defendant_id']
                    
                mape_per_defendant = np.mean(
                    np.abs(predictions - y[:, None]),
                    axis=1
                )
                save_to_csv_by_fold(mape_per_defendant, 'mape', 'MAPE_per_defendant', instability_type, model, tests[i], fold_id, defendant_ids)
                print(f'[{model}] Individual MAPE values: ')
                print(mape_per_defendant)
                
                mean_mape = np.mean(mape_per_defendant)
                save_to_csv_all_folds(mean_mape, 'mape', 'Mean_MAPE', instability_type, model, tests[i], fold_id)          
                print(f'[{model}] Mean MAPE: ')
                print(mean_mape)
        
# Instability metric
def top_k_jaccard(instability_type, tests, folds):
    for i in range(2):
        for model in models:
            for fold_id in folds:
                predictions_with_id_df = pd.read_csv(
                    f'predictions/{instability_type}/'
                    f'{model}_Predictions_{tests[i]}_{fold_id}.csv'
                )
                predictions = predictions_with_id_df.drop(columns=['defendant_id']).to_numpy()
                
                k = int(0.10 * len(predictions))
                jaccard_scores = top_k_jaccard_all(predictions, k)
                save_to_csv_by_fold(jaccard_scores, 'jaccard_scores', 'Jaccard_Scores', instability_type, model, tests[i], fold_id)
                
                mean_jaccard = np.mean(jaccard_scores)
                save_to_csv_by_fold(mean_jaccard, 'jaccard_scores', 'Mean_Jaccard', instability_type, model, tests[i], fold_id)
                print("Mean Top-K Jaccard:", mean_jaccard)
                                                
                std_jaccard = np.std(jaccard_scores)
                save_to_csv_by_fold(std_jaccard, 'jaccard_scores', 'STD_Jaccard', instability_type, model, tests[i], fold_id)
                print("SD:", std_jaccard)
        
# Instability metric
def classification_instability_index(instability_type, tests, folds):
    for i in range(2):
        for model in models:
            for fold_id in folds:
                predictions_with_id_df = pd.read_csv(
                    f'predictions/{instability_type}/'
                    f'{model}_Predictions_{tests[i]}_{fold_id}.csv'
                )
                predictions = predictions_with_id_df.drop(columns=['defendant_id']).to_numpy()
                defendant_ids = predictions_with_id_df['defendant_id']
                
                threshold = 0.5
                
                # Convert to binary labels
                labels = (predictions >= threshold).astype(int)
            
                # Compute instability per individual
                majority_label = np.round(labels.mean(axis=1)).astype(int)
            
                disagreements = (labels != majority_label[:, None]).sum(axis=1)
                cii_individual = disagreements / labels.shape[1]
                save_to_csv_by_fold(cii_individual, 'cii', 'CII_per_defendant', instability_type, model, tests[i], fold_id, defendant_ids)
                                
                # Mean CII
                cii_mean = np.mean(cii_individual)
                save_to_csv_all_folds(cii_mean, 'cii', 'CII_Mean', instability_type, model, tests[i], fold_id)
                print(f'{model} CII: {cii_mean:.4f}')
                
                plt.hist(cii_individual, bins=20)
                plt.xlabel('CII per Individual')
                plt.ylabel('Count')
                plt.title(f'Classification Instability Distribution ({instability_type} - {model}, Fold {fold_id})')
                plt.savefig(
                    f'metrics/{instability_type}/cii/'
                    f'{model}_{tests[i]}_CII_Distribution_Plot_{fold_id}.png')
                plt.close()

# Performance metric
def calibration_plot(y, instability_type, tests, folds):
    for i in range(2):
        for model in models:
            for fold_id in folds:
                predictions_with_id_df = pd.read_csv(
                    f'predictions/{instability_type}/'
                    f'{model}_Predictions_{tests[i]}_{fold_id}.csv'
                )
                predictions = predictions_with_id_df.drop(
                    columns=['defendant_id']
                ).to_numpy()

                plt.figure(figsize=(8, 6))

                for run in range(predictions.shape[1]):
                    prob_true, prob_pred = calibration_curve(
                        y[i],
                        predictions[:, run],
                        n_bins=10,
                        strategy='uniform'
                    )

                    plt.plot(
                        prob_pred,
                        prob_true,
                        alpha=0.1
                    )

                plt.plot(
                    [0, 1],
                    [0, 1],
                    linestyle='--',
                    label='Perfect Calibration'
                )

                plt.xlabel('Mean Predicted Probability')
                plt.ylabel('Fraction of Positives')
                plt.title(
                    f'{instability_type} - {model} - {tests[i]} - '
                    f'Calibration Plot - Fold {fold_id}'
                )
                plt.legend()
                plt.tight_layout()

                plt.savefig(
                    f'metrics/{instability_type}/calibration/'
                    f'{model}_{tests[i]}_Calibration_Plot_{fold_id}.png',
                    dpi=300,
                    bbox_inches='tight'
                )

                plt.close()
                
                
def demographic_false_positive_rate(y, instability_type, tests, folds):
    fpr_by_group = {}

    for group in demographics.unique():
        mask = demographics == group

        tn, fp, fn, tp = confusion_matrix(
            y_true[mask],
            y_pred[mask],
            labels=[0, 1]
        ).ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        fpr_by_group[group] = fpr