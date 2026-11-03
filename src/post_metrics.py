import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, r2_score

from itertools import combinations
from statsmodels.nonparametric.smoothers_lowess import lowess

import directory as dir
import csv_actions as csva

# Constant Variables
DFPR_THRESHOLD = 0.5

RACE_NAMES = [
    'race:African-American',
    'race:Asian',
    'race:Caucasian',
    'race:Native_American',
    'race:Hispanic',
    'race:Other'
]
MODEL_NAMES = [ 'LR', 'RF', 'LSVC', 'XGB', 'MLP' ]
EVAL_NAMES = [ 'Validation', 'Test'] 

class PostMetrics:
    def __init__(
        self, 
        X_train, y_train,   # Training data (not split)
        X_val_list, y_val_list, val_defendant_ids_list,
        X_test, y_test, test_defendant_ids,     # Testing data  
        no_of_folds,        
        predictions_dir,  
        instability_type,
        force_reset
    ):
        self.force_reset = force_reset
        
        # Official Training data (not split yet to validation and bootstrapped samples)
        self.X_train = X_train
        self.y_train = y_train
        
        # Official Testing Data
        self.X_test = X_test
        self.y_test = y_test
        self.test_defendant_ids = test_defendant_ids
        
        # Validation Data Arrays
        self.X_val_list = X_val_list
        self.y_val_list = y_val_list
        self.val_defendant_ids_list = val_defendant_ids_list
        
        # Directory for predictions
        self.predictions_dir = predictions_dir
        self.instability_type = instability_type
        
        # Consistent Data
        self.no_of_folds = no_of_folds
        
        # Directory for metrics
        self.metrics_dir = f'metrics/{self.instability_type}'
        if self.force_reset:
            self.delete_dirs(self.metrics_dir)
    
    # Deleting a directory
    def delete_dirs(self, directory_name):
        try:
            shutil.rmtree(directory_name, ignore_errors=True)
        except Exception as e:
            print(f"An error occurred: ", e)
    
    def print_metric_progress(
        self, eval_name, model_name, fold_id, metric_name=None,
        completed=None, total=None
    ):
        progress = f'{eval_name}, {model_name}, Fold {fold_id}/{self.no_of_folds}'
        if metric_name is not None:
            if completed is None or total is None:
                progress += f' - Getting {metric_name}...'
            else:
                progress += (
                    f' - Getting {metric_name}s... '
                    f'{round((completed / total) * 100)}%'
                )

        if completed == 100: 
            print(progress)
        else:
            print(progress, end='\r')
    
    # ================
    # GETTER FUNCTIONS
    # ================
    
    # Gets the defendant ids from the test or validation data
    def get_defendant_ids(self, eval_name, fold_id=None):
        return (
            self.test_defendant_ids
            if eval_name == 'Test'
            else self.val_defendant_ids_list[fold_id-1]
        )
    
    # Gets the y_values from the test or validation data
    def get_y_values(self, eval_name, fold_id=None):        
        return (
            self.y_test 
            if eval_name == 'Test' 
            else self.y_val_list[fold_id-1]
        )
    
    # Gets the eval (test or validation) values requested
    def get_eval_values(self, eval_name, fold_id=None):
        return (
            self.X_test 
            if eval_name == 'Test' 
            else self.X_val_list[fold_id-1]
        )
    
    # Gets the predictions per fold
    def get_predictions(self, model_name, eval_name, fold_id):
        return pd.read_csv(f'{self.predictions_dir}/{model_name}_Predictions_{eval_name}_{fold_id}.csv')                           
    
    # Gets the shap values
    def get_shap_values(self, shap_values_dir, model_name, eval_name, fold_id):
        csv_path = (
            f'{shap_values_dir}/SHAP_{model_name}_'
            f'{eval_name}_Fold_{fold_id}.csv'
        )
        shap_values = pd.read_csv(csv_path)
        return shap_values.rename(columns={
            'defendant_id': 'Defendant_ID',
            'feature': 'Feature',
        })
    
    # Returns the csv file path
    def get_csv_path(
        self, metric_dir, model_name, metric_name, eval_name, custom_title=None, fold_id=None,
    ):
        return (
            f'{metric_dir}/{custom_title if custom_title != None else 'Summary'}_{metric_name}_{model_name}_{self.instability_type}_{eval_name}.csv'
            if fold_id == None
            else f'{metric_dir}/{f'{custom_title}_' if custom_title != None else ''}{metric_name}_{model_name}_{self.instability_type}_{eval_name}_Fold_{fold_id}.csv'
        )
    
    
    # =======
    # METRICS
    # =======
    # Performance metric
    def roc_auc(self):
        print(f'Metric: ROC-AUC')
        metric_name = 'ROC_AUC'
        
        for eval_name in EVAL_NAMES:
            for model_name in MODEL_NAMES:
                roc_auc_dir = f'{self.metrics_dir}/{metric_name}/{model_name}'
                dir.create_nested(roc_auc_dir)
                
                summary_dict = {
                    'Fold': range(1, self.no_of_folds + 1),
                    f'Mean_{metric_name}': [],
                    f'STD_{metric_name}': [],
                }
                
                for fold_id in range(1, self.no_of_folds + 1):
                    predictions = self.get_predictions(model_name, eval_name, fold_id)
                    y_eval = self.get_y_values(eval_name, fold_id)
                    
                    fold_csv_path = self.get_csv_path(roc_auc_dir, model_name, metric_name, eval_name, fold_id=fold_id)
                    csva.save_to_csv(fold_csv_path, 'Run', predictions.columns)
                    
                    roc_aucs = []
                    for i, col in enumerate(predictions.columns, start=1):
                        self.print_metric_progress(eval_name, model_name, fold_id, 'ROC-AUC', i, len(predictions.columns))
                        y_pred_proba = predictions[col]
                        
                        roc_auc = roc_auc_score(y_eval, y_pred_proba)
                        roc_aucs.append(roc_auc)    
                        
                    csva.save_to_csv(fold_csv_path, 'ROC_AUC_Score', roc_aucs)
                
                    summary_dict[f'Mean_{metric_name}'].append(np.mean(roc_aucs))
                    summary_dict[f'STD_{metric_name}'].append(np.std(roc_aucs, ddof=1))
                    
                summary_csv_path = self.get_csv_path(roc_auc_dir, model_name, metric_name, eval_name)
                csva.save_to_csv(summary_csv_path, summary_dict)
                print()
        
    # Performance metric
    def brier_score(self):
        print('Metric: Brier Score')
        metric_name = 'Brier_Score'
        
        for eval_name in EVAL_NAMES:            
            for model_name in MODEL_NAMES:         
                brier_score_dir = f'{self.metrics_dir}/{metric_name}/{model_name}'
                dir.create_nested(brier_score_dir)
                
                summary_dict = {
                    'Fold': range(1, self.no_of_folds + 1),
                    f'Mean_{metric_name}': [],
                    f'STD_{metric_name}': [],
                }
                
                for fold_id in range(1, self.no_of_folds + 1):
                    y_eval = self.get_y_values(eval_name, fold_id)             
                    predictions = self.get_predictions(model_name, eval_name, fold_id=fold_id)
                    
                    fold_csv_path = self.get_csv_path(brier_score_dir, model_name, metric_name, eval_name, fold_id=fold_id)
                    csva.save_to_csv(fold_csv_path, 'Run', predictions.columns)
                    
                    brier_score_list = []
                    for i, col in enumerate(predictions.columns, start=1):
                        self.print_metric_progress(eval_name, model_name, fold_id, 'Brier Score', i, len(predictions.columns))
                        y_pred_proba = predictions[col]
                        
                        brier_score = brier_score_loss(y_eval, y_pred_proba)
                        brier_score_list.append(brier_score)
                        
                    csva.save_to_csv(fold_csv_path, metric_name, brier_score_list)
                    
                    summary_dict[f'Mean_{metric_name}'].append(np.mean(brier_score_list))
                    summary_dict[f'STD_{metric_name}'].append(np.std(brier_score_list, ddof=1))
                
                summary_csv_path = self.get_csv_path(brier_score_dir, model_name, metric_name, eval_name)
                csva.save_to_csv(summary_csv_path, summary_dict)
                print()

    # Instability metric
    def ninety_five_prediction_error(self):
        print('Metric: 95 Prediction Error')
        metric_name = '95_Prediction_Error'
        
        for eval_name in EVAL_NAMES:
            for model_name in MODEL_NAMES:
                si_dir = f'{self.metrics_dir}/{metric_name}/{model_name}'
                dir.create_nested(si_dir)

                summary_csv_path = self.get_csv_path(si_dir, model_name, metric_name, eval_name)
                summary_dict = {
                    'Fold': [],
                    'Lower_95_Error': [],
                    'Upper_95_Error': [],
                    'Mean_Error': [],
                    'Median_Error': [],
                }
                all_fold_errors = []

                for fold_id in range(1, self.no_of_folds + 1):
                    self.print_metric_progress(eval_name, model_name, fold_id, '95% Prediction Error', 100)
                    predictions = self.get_predictions(model_name, eval_name, fold_id)
                    y_eval = self.get_y_values(eval_name, fold_id)

                    errors = np.abs(y_eval[:, np.newaxis] - predictions)
                    fold_errors = errors.to_numpy().flatten()
                    
                    all_fold_errors.append(fold_errors)
                    summary_dict['Fold'].append(fold_id)
                    summary_dict['Lower_95_Error'].append(np.percentile(fold_errors, 2.5))
                    summary_dict['Upper_95_Error'].append(np.percentile(fold_errors, 97.5))
                    summary_dict['Mean_Error'].append(np.mean(fold_errors))
                    summary_dict['Median_Error'].append(np.median(fold_errors))
                    print()
                
                # Box plot
                plt.figure(figsize=(12, 6))
                plt.boxplot(
                    all_fold_errors,
                    tick_labels=[
                        f'Fold {fold_id}'
                        for fold_id in range(1, self.no_of_folds + 1)
                    ],
                    notch=True
                )

                # Zero = no prediction error
                plt.axhline(
                    y=0,
                    linestyle='--',
                    linewidth=1
                )

                plt.xlabel('Fold')
                plt.ylabel('Prediction Error (y - predicted probability)')
                plt.title(f'{self.instability_type} - {model_name} - 95% Prediction Error Distribution - {eval_name}')
                plt.tight_layout()
                plot_path = summary_csv_path.replace('.csv', '_Boxplot.png')
                plt.savefig(plot_path)
                plt.close()
                
                csva.save_to_csv(summary_csv_path, summary_dict)
        
    # Instability metric
    def mean_absolute_prediction_error(self): 
        print('Metric: Mean Absolute Prediction Error (MAPE)')
        metric_name = 'MAPE'
            
        for eval_name in EVAL_NAMES:
            for model_name in MODEL_NAMES:
                mape_dir = f'{self.metrics_dir}/{metric_name}/{model_name}'
                dir.create_nested(mape_dir)

                summary_dict = {
                    'Fold': [f'Fold_{i}' for i in range(1, self.no_of_folds + 1)],
                    'Mean_MAPE': [],
                }
                
                for fold_id in range(1, self.no_of_folds + 1):
                    self.print_metric_progress(eval_name, model_name, fold_id, 'Mean Absolute Prediction Error', 100)
                    predictions = self.get_predictions(model_name, eval_name, fold_id=fold_id)
                        
                    mape_per_defendant = predictions.sub(
                        predictions.mean(axis=1), 
                        axis=0
                    ).abs().mean(axis=1)
                    
                    fold_csv_path = self.get_csv_path(mape_dir, model_name, metric_name, eval_name, fold_id=fold_id)
                    csva.save_to_csv(fold_csv_path, 'Defendant_ID', self.get_defendant_ids(eval_name, fold_id))
                    csva.save_to_csv(fold_csv_path, 'MAPE', mape_per_defendant)
                    
                    summary_dict['Mean_MAPE'].append(np.mean(mape_per_defendant))
                    print()
                    
                summary_csv_path = self.get_csv_path(mape_dir, model_name, metric_name, eval_name)
                csva.save_to_csv(summary_csv_path, summary_dict)
            
    # Instability metric 
    def top_k_jaccard(self):
        print('Metric: Top-K Jaccard')
        metric_name = 'Top_K_Jaccard'
        
        for eval_name in EVAL_NAMES:            
            for model_name in MODEL_NAMES:
                tk_jaccard_dir = f'{self.metrics_dir}/{metric_name}/{model_name}'
                dir.create_nested(tk_jaccard_dir)

                summary_dict = {
                    'Fold': range(1, self.no_of_folds + 1),
                    f'Mean_{metric_name}': [],
                    f'STD_{metric_name}': [],
                }

                for fold_id in range(1, self.no_of_folds + 1):          
                    self.print_metric_progress(eval_name, model_name, fold_id, 'Top-K Jaccard', 100)     
                    jaccard_fold_dict = {
                        'Run_1': [],
                        'Run_2': [],
                        'Jaccard': []
                    }

                    predictions = self.get_predictions(model_name, eval_name, fold_id=fold_id)
                        
                    k = max(1, int(0.10 * len(predictions)))
                    for col1, col2 in combinations(predictions.columns, 2):
                        jaccard_fold_dict['Run_1'].append(col1)
                        jaccard_fold_dict['Run_2'].append(col2)
                        
                        run1 = predictions[col1]
                        run2 = predictions[col2]
                        
                        top_k_1 = run1.nlargest(k).index
                        top_k_2 = run2.nlargest(k).index
                        
                        intersection = len(set(top_k_1) & set(top_k_2))
                        union = len(set(top_k_1) | set(top_k_2))

                        jaccard = intersection / union
                        jaccard_fold_dict['Jaccard'].append(jaccard)
                        
                    fold_csv_path = self.get_csv_path(tk_jaccard_dir, model_name, metric_name, eval_name, fold_id=fold_id)
                    csva.save_to_csv(fold_csv_path, jaccard_fold_dict)
                        
                    summary_dict[f'Mean_{metric_name}'].append(np.mean(jaccard_fold_dict['Jaccard']))
                    summary_dict[f'STD_{metric_name}'].append(np.std(jaccard_fold_dict['Jaccard'], ddof=1))
                    print()
                    
                summary_csv_path = self.get_csv_path(tk_jaccard_dir, model_name, metric_name, eval_name)
                csva.save_to_csv(summary_csv_path, summary_dict)
                
            
    # Instability metric 
    def classification_instability_index(self):
        print('Metric: Classification Instability Index (CII)')
        threshold = 0.5
        metric_name = 'CII'

        for eval_name in EVAL_NAMES:
            for model_name in MODEL_NAMES:
                cii_dir = f'{self.metrics_dir}/{metric_name}/{model_name}'
                dir.create_nested(cii_dir)

                summary_dict = {
                    'Fold': range(1, self.no_of_folds + 1),
                    f'Mean_{metric_name}': [],
                    f'STD_{metric_name}': [],
                }
                
                for fold_id in range(1, self.no_of_folds + 1):
                    self.print_metric_progress(eval_name, model_name, fold_id, 'Classification Instability Index', 100)
                    predictions = self.get_predictions(model_name, eval_name, fold_id=fold_id)
                    
                    labels = (predictions >= threshold).astype(int)
                    n_1 = labels.sum(axis=1)
                    n_0 = labels.shape[1] - n_1
                    flips = n_0 * n_1
                    
                    fold_csv_path = self.get_csv_path(cii_dir, model_name, metric_name, eval_name, fold_id=fold_id)
                    
                    defendant_ids = self.get_defendant_ids(eval_name, fold_id)
                    csva.save_to_csv(fold_csv_path, 'Defendant_ID', defendant_ids)
                    
                    cii_individual = flips / (labels.shape[1] * (labels.shape[1] - 1) / 2)
                    csva.save_to_csv(fold_csv_path, 'CII', cii_individual)
                                  
                    summary_dict[f'Mean_{metric_name}'].append(np.mean(cii_individual))
                    summary_dict[f'STD_{metric_name}'].append(np.std(cii_individual, ddof=1))
                    
                    plt.hist(cii_individual, bins=20)
                    plt.xlabel('CII per Individual')
                    plt.ylabel('Count')
                    plt.title(f'Classification Instability Distribution ({self.instability_type} - {model_name}, Fold {fold_id})')
                    plt.savefig(fold_csv_path.replace('csv', 'png'))
                    plt.close()
                    print()
                    
                summary_csv_path = self.get_csv_path(cii_dir, model_name, metric_name, eval_name)
                csva.save_to_csv(summary_csv_path, summary_dict)
                

    # Performance metric 
    def calibration_plot(self):
        print('Metric: Calibration Plot')
        
        metric_name = 'Calibration_Plot'
        for eval_name in EVAL_NAMES:
            for model_name in MODEL_NAMES:
                cal_plot_dir = f'{self.metrics_dir}/{metric_name}/{model_name}'
                dir.create_nested(cal_plot_dir)

                r2_summary_records = []
                for fold_id in range(1, self.no_of_folds + 1):                    
                    y_eval = np.asarray(self.get_y_values(eval_name, fold_id)).ravel()
                    predictions = self.get_predictions(model_name, eval_name, fold_id=fold_id)

                    plt.figure(figsize=(8, 6))
                    
                    r2_scores = []
                    for i, col in enumerate(predictions.columns, start=1):
                        self.print_metric_progress(eval_name, model_name, fold_id, 'Calibration Plot', i, len(predictions.columns))
                        y_pred_proba = predictions[col]

                        prob_true, prob_pred = calibration_curve(
                            y_eval,
                            y_pred_proba,
                            n_bins=10,
                            strategy='uniform'
                        )

                        plt.plot(
                            prob_pred,
                            prob_true,
                            alpha=0.1
                        )

                        run_r2 = (
                            r2_score(y_eval, y_pred_proba)
                            if np.unique(y_eval).size > 1
                            else np.nan
                        )
                        r2_scores.append(run_r2)

                    plt.plot(
                        [0, 1],
                        [0, 1],
                        linestyle='--',
                        label='Perfect Calibration'
                    )

                    plt.xlabel('Mean Predicted Probability')
                    plt.ylabel('Fraction of Positives')
                    plt.title(
                        f'{self.instability_type} - {model_name} - {eval_name} - '
                        f'Calibration Plot - Fold {fold_id}'
                    )

                    valid_r2_scores = np.asarray(r2_scores, dtype=float)
                    valid_r2_scores = valid_r2_scores[~np.isnan(valid_r2_scores)]
                    mean_r2 = (
                        float(np.mean(valid_r2_scores))
                        if len(valid_r2_scores) else np.nan
                    )
                    
                    std_r2 = (
                        float(np.std(valid_r2_scores, ddof=1))
                        if len(valid_r2_scores) > 1 else np.nan
                    )
                    
                    if np.isfinite(mean_r2) and np.isfinite(std_r2):
                        r2_label = (
                            'Outcome $R^2$ vs prevalence '
                            '(mean $\\pm$ SD across runs): '
                            f'{mean_r2:.3f} $\\pm$ {std_r2:.3f}'
                        )
                    elif np.isfinite(mean_r2):
                        r2_label = (
                            'Outcome $R^2$ vs prevalence '
                            f'(mean across runs): {mean_r2:.3f}'
                        )
                    else:
                        r2_label = 'Outcome $R^2$ undefined'
                        
                    plt.text(
                        0.05, 0.95,
                        r2_label,
                        transform=plt.gca().transAxes,
                        verticalalignment='top',
                        bbox=dict(facecolor='white', alpha=0.8)
                    )

                    plt.legend()
                    plt.tight_layout()

                    img_file_path = (
                        f'{cal_plot_dir}/{metric_name}_{model_name}_'
                        f'{self.instability_type}_{eval_name}_Fold_{fold_id}.png'
                    )
                    
                    plt.savefig(
                        img_file_path,
                        dpi=300,
                        bbox_inches='tight'
                    )
                    plt.close()

                    per_run_r2_path = self.get_csv_path(cal_plot_dir, model_name, metric_name, eval_name, custom_title='R2', fold_id=fold_id)
                    r2_outcome_dict = pd.DataFrame({
                        'Run': predictions.columns,
                        'Outcome_R2_vs_Prevalence': r2_scores,
                    })
                    csva.save_to_csv(per_run_r2_path, r2_outcome_dict)
                    
                    r2_summary_records.append({
                        'Fold': fold_id,
                        'Mean_Outcome_R2_vs_Prevalence': mean_r2,
                        'SD_Outcome_R2_vs_Prevalence': std_r2,
                        'Valid_Runs': len(valid_r2_scores),
                    })

                r2_summary_path = self.get_csv_path(cal_plot_dir, model_name, metric_name, eval_name, custom_title='R2_Summary')
                pd.DataFrame(r2_summary_records).to_csv(
                    r2_summary_path,
                    index=False
                )
                print()
                            
    # Fairness metric: subgroup false positive rates and their max-min gap.
    def demographic_false_positive_rate(self):
        print('Metric: Race-specific FPR and Maximum FPR Difference (DFPR)')
        metric_name = 'DFPR'
        
        for eval_name in EVAL_NAMES:
            for model_name in MODEL_NAMES:
                dfpr_dir = f'{self.metrics_dir}/{metric_name}/{model_name}'
                dir.create_nested(dfpr_dir)

                mean_values_by_fold = {}
                std_values_by_fold = {}

                for fold_id in range(1, self.no_of_folds + 1):
                    eval_df = self.get_eval_values(eval_name, fold_id)
                    y_eval = np.asarray(self.get_y_values(eval_name, fold_id)).ravel()
                    predictions = self.get_predictions(model_name, eval_name, fold_id=fold_id)

                    # Get the indices of defendants under each race
                    race_masks = { race: (eval_df[race].to_numpy() == 1) for race in RACE_NAMES }
                    
                    # Get the is_recid = 0 count for every race
                    not_recid_count = { race: int(np.sum(y_eval[mask] == 0)) for race, mask in race_masks.items() }
                    
                    # Init rates by race dict
                    rates_by_race = { race: [] for race in RACE_NAMES }
                    
                    disparities_by_run = {
                        'Run': [ col for col in predictions.columns ] ,
                        'Max-Min Delta FPR': [],
                        'Groups Compared': []
                    }
                    for i, col in enumerate(predictions.columns, start=1):
                        self.print_metric_progress(eval_name, model_name, fold_id, 'DFPR', i, len(predictions.columns))
                        y_pred_proba = predictions[col].to_numpy() >= DFPR_THRESHOLD
                        
                        rates_by_run = {}
                        for race in RACE_NAMES:
                            mask = race_masks[race]

                            y_group = y_eval[mask]
                            pred_group = y_pred_proba[mask]
                            actual_negative = y_group == 0
                            negative_count = int(np.sum(actual_negative))

                            dfpr = (
                                float(np.mean(pred_group[actual_negative]))
                                if negative_count > 0
                                else np.nan
                            )
                            rates_by_race[race].append(dfpr)
                            
                            if not np.isnan(dfpr):
                                rates_by_run[race] = dfpr

                        delta_fpr = (
                            max(rates_by_run.values()) - min(rates_by_run.values())
                            if len(rates_by_run) >= 2
                            else np.nan
                        )
                            
                        disparities_by_run['Max-Min Delta FPR'].append(delta_fpr)
                        disparities_by_run['Groups Compared'].append(len(rates_by_run))

                    delta_csv_path = self.get_csv_path(dfpr_dir, model_name, metric_name, eval_name, custom_title='Delta_FPR', fold_id=fold_id)
                    pd.DataFrame(disparities_by_run).to_csv(
                        delta_csv_path,
                        index=False
                    )
                    
                    rates_dict = {
                        'Race': RACE_NAMES,
                        'Non-Recidivists': [not_recid_count[race] for race in RACE_NAMES],
                        **{
                            col: [rates_by_race[race][i] for race in RACE_NAMES]
                            for i, col in enumerate(predictions.columns)
                        },
                    }
                    fold_csv_path = self.get_csv_path(dfpr_dir, model_name, metric_name, eval_name, fold_id=fold_id)
                    pd.DataFrame(rates_dict).to_csv(fold_csv_path, index=False)

                    fold_mean_list = {}
                    fold_std_list = {}
                    for race in RACE_NAMES:
                        valid_scores = np.asarray(rates_by_race[race], dtype=float)
                        valid_scores = valid_scores[~np.isnan(valid_scores)]
                        
                        fold_mean_list[race] = float(np.mean(valid_scores)) if len(valid_scores) else np.nan                        
                        fold_std_list[race] = float(np.std(valid_scores, ddof=1)) if len(valid_scores) > 1 else np.nan

                    valid_disparities = np.asarray(
                        disparities_by_run['Max-Min Delta FPR'],
                        dtype=float
                    )
                    valid_disparities = valid_disparities[~np.isnan(valid_disparities)]
                    
                    fold_mean_list['Max-Min Delta FPR'] = (
                        float(np.mean(valid_disparities))
                        if len(valid_disparities) else np.nan
                    )
                    fold_std_list['Max-Min Delta FPR'] = (
                        float(np.std(valid_disparities, ddof=1))
                        if len(valid_disparities) > 1 else np.nan
                    )
                    
                    mean_values_by_fold[f'Fold_{fold_id}'] = fold_mean_list
                    std_values_by_fold[f'Fold_{fold_id}'] = fold_std_list

                summary_rows = [*RACE_NAMES, 'Max-Min Delta FPR']
                mean_dict = {'Race': summary_rows}
                std_dict = {'Race': summary_rows}
                
                for fold_id in range(1, self.no_of_folds + 1):
                    fold_column = f'Fold_{fold_id}'
                    mean_dict[fold_column] = [
                        mean_values_by_fold[fold_column][race]
                        for race in summary_rows
                    ]
                    std_dict[fold_column] = [
                        std_values_by_fold[fold_column][race]
                        for race in summary_rows
                    ]

                mean_csv_path = self.get_csv_path(dfpr_dir, model_name, metric_name, eval_name, custom_title='Means')
                pd.DataFrame(mean_dict).to_csv(mean_csv_path, index=False)
                
                std_csv_path = self.get_csv_path(dfpr_dir, model_name, metric_name, eval_name, custom_title='STD')
                pd.DataFrame(std_dict).to_csv(std_csv_path, index=False)
                print()
                
                
    # SHAP ANALYSIS
    # Uses the runtime shap values per fold
    def shap_analysis(self):
        print('Metric: SHAP Analysis')
        metric_name = 'SHAP_Analysis'
        
        for eval_name in EVAL_NAMES:
            for model_name in MODEL_NAMES:
                shap_values_dir = f'{self.metrics_dir}/shap_values/{model_name}'
                shap_dir = f'{self.metrics_dir}/{metric_name}/{model_name}'
                dir.create_nested(shap_dir)

                for fold_id in range(1, self.no_of_folds + 1):
                    self.print_metric_progress(eval_name, model_name, fold_id, 'SHAP', 100)
                    shap_values = self.get_shap_values(shap_values_dir, model_name, eval_name, fold_id)
                    
                    shap_columns = shap_values.columns
                    shap_valid_columns = shap_columns.drop(
                        ['Defendant_ID', 'Feature'],
                        errors='ignore'
                    )
                    shap_mean = shap_values[shap_valid_columns].mean(axis=1)

                    output = {
                        'Defendant_ID': shap_values['Defendant_ID'],
                        'Feature': shap_values['Feature'],
                        'SHAP_Mean': shap_mean
                    }
                    
                    shap_csv_path = self.get_csv_path(shap_dir, model_name, metric_name, eval_name, fold_id=fold_id)
                    csva.save_to_csv(csv_path=shap_csv_path, data=output)    
        print()
