import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap

from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    brier_score_loss, 
    mean_squared_error
)

from itertools import combinations

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
            dir.clear_nested(self.metrics_dir)
    
    # Prints the progress of a metric
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
        csv_path = f'{self.predictions_dir}/{model_name}_Predictions_{eval_name}_{fold_id}.csv'
        return csva.load_csv_data(csv_path)                       
    
    # Gets the shap values
    def get_shap_values(self, shap_values_dir, model_name, eval_name, fold_id):
        csv_path = f'{shap_values_dir}/SHAP_{model_name}_{eval_name}_Fold_{fold_id}.csv'
        shap_values = csva.load_csv_data(csv_path)
        return shap_values.rename(columns={
            'defendant_id': 'Defendant_ID',
            'feature': 'Feature',
        })
    
    # Returns the csv file path
    def generate_csv_path(
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
    
    """
    🗹 Receiver Operating Characteristic - Area Under the Curve (ROC-AUC)
    - Performance Metric
    - AUC gives you an overall idea of how well your model is doing at sorting positives and negatives, 
      without being affected by the threshold you set for classification. 
    - A higher AUC means your model is doing good.
    """
    def roc_auc(self):
        print(f'[Metric] ROC-AUC')
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
                    y_eval = np.asarray(self.get_y_values(eval_name, fold_id)).ravel()
                    
                    fold_csv_path = self.generate_csv_path(roc_auc_dir, model_name, metric_name, eval_name, fold_id=fold_id)
                    csva.save_to_csv(fold_csv_path, 'Run', predictions.columns)
                    
                    roc_aucs = []
                    tpr_list = []

                    # Common FPR values for all runs
                    mean_fpr = np.linspace(0, 1, 100)
                    for i, col in enumerate(predictions.columns, start=1):
                        self.print_metric_progress(eval_name, model_name, fold_id, 'ROC-AUC', i, len(predictions.columns))
                        y_pred_proba = predictions[col]
                        
                        roc_auc = roc_auc_score(y_eval, y_pred_proba)
                        roc_aucs.append(roc_auc)    
                        
                        # Plotting the ROC-AUC 
                        fpr, tpr, _ = roc_curve(y_eval, y_pred_proba)
                        
                        # Interpolate TPR to common FPR values
                        interp_tpr = np.interp(mean_fpr, fpr, tpr)
                        interp_tpr[0] = 0.0
                        tpr_list.append(interp_tpr)
                        
                    csva.save_to_csv(fold_csv_path, 'ROC_AUC_Score', roc_aucs)

                    mean_auc = np.mean(roc_aucs)
                    std_auc = np.std(roc_aucs, ddof=1)
                    summary_dict[f'Mean_{metric_name}'].append(mean_auc)
                    summary_dict[f'STD_{metric_name}'].append(std_auc)
                    
                    # ROC_AUC Curve
                    mean_tpr = np.mean(tpr_list, axis=0)
                    mean_tpr[-1] = 1.0
                
                    plt.plot(mean_fpr, mean_tpr, label=f'Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})')
                    plt.plot([0, 1], [0, 1], '--', label='Random Classifier')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(
                        f'{self.instability_type} - {model_name} - '
                        f'{eval_name} - ROC Curve - Fold {fold_id}'
                    )
                    plt.legend()
                    plt.tight_layout()
                    
                    img_file_path = (
                        f'{roc_auc_dir}/{metric_name}_{model_name}_'
                        f'{self.instability_type}_{eval_name}_Fold_{fold_id}_{col}.png'
                    )
                    plt.savefig(
                        img_file_path,
                        dpi=300,
                        bbox_inches='tight'
                    )
                    plt.close()
                    
                summary_csv_path = self.generate_csv_path(roc_auc_dir, model_name, metric_name, eval_name)
                csva.save_to_csv(summary_csv_path, data=summary_dict)
                print()
        
    """
    🗹 Brier Score
    - Performance metric
    - Brier Score can take on any value between 0 and 1, with 0 being the best score 
      achievable and 1 being the worst score achievable. 
    - The lower the Brier Score, the more accurate the prediction(s).
    """
    def brier_score(self):
        print('[Metric] Brier Score')
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
                    y_eval = np.asarray(self.get_y_values(eval_name, fold_id)).ravel()             
                    predictions = self.get_predictions(model_name, eval_name, fold_id=fold_id)
                    
                    fold_csv_path = self.generate_csv_path(brier_score_dir, model_name, metric_name, eval_name, fold_id=fold_id)
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
                
                summary_csv_path = self.generate_csv_path(brier_score_dir, model_name, metric_name, eval_name)
                csva.save_to_csv(summary_csv_path, data=summary_dict)
                print()
    
    """
    🗹 Calibration Curve
    - Performance metric
    - Calibration curves are used to evaluate how calibrated a classifier is i.e., how the probabilities of predicting each class label differ. 
      The x-axis represents the average predicted probability in each bin. 
      The y-axis is the ratio of positives (the proportion of positive predictions). 
    - The curve of the ideal calibrated model is a linear straight line from (0, 0) moving linearly. 
    """
    def calibration_curve(self):
        print('[Metric] Calibration Curve')
        metric_name = 'Calibration_Curve'

        for eval_name in EVAL_NAMES:
            pooled_predictions_by_fold = {
                fold_id: {}
                for fold_id in range(1, self.no_of_folds + 1)
            }

            for model_name in MODEL_NAMES:
                cal_curve_dir = f'{self.metrics_dir}/{metric_name}/{model_name}'
                dir.create_nested(cal_curve_dir)

                mse_summary_records = []
                for fold_id in range(1, self.no_of_folds + 1):
                    y_eval = np.asarray(self.get_y_values(eval_name, fold_id)).ravel()
                    predictions = self.get_predictions(model_name, eval_name, fold_id=fold_id)

                    # Pool all samples from all runs into one curve per model/fold.
                    pooled_predictions_by_fold[fold_id][model_name] = (
                        np.repeat(y_eval, predictions.shape[1]),
                        predictions.to_numpy(dtype=float).ravel(),
                    )

                    mse_scores = []
                    for i, col in enumerate(predictions.columns, start=1):
                        self.print_metric_progress(eval_name, model_name, fold_id, 'Calibration Curve', i, len(predictions.columns))
                        y_pred_proba = predictions[col]

                        # Preserve per-run probability error summaries.
                        run_mse = (
                            mean_squared_error(y_eval, y_pred_proba)
                            if np.unique(y_eval).size > 1
                            else np.nan
                        )
                        mse_scores.append(run_mse)

                    mean_mse = np.nanmean(mse_scores)
                    std_mse = np.nanstd(mse_scores, ddof=1)

                    mse_label = 'Outcome_MSE'
                    per_run_mse_path = self.generate_csv_path(cal_curve_dir, model_name, metric_name, eval_name, custom_title='MSE', fold_id=fold_id)
                    mse_outcome_dict = {
                        'Run': predictions.columns,
                        mse_label: mse_scores,
                    }
                    csva.save_to_csv(per_run_mse_path, data=mse_outcome_dict)
                    
                    mse_summary_records.append({
                        'Fold': fold_id,
                        f'Mean_{mse_label}': mean_mse,
                        f'SD_{mse_label}': std_mse
                    })

                mse_summary_path = self.generate_csv_path(cal_curve_dir, model_name, metric_name, eval_name, custom_title='MSE_Summary')
                csva.save_to_csv(mse_summary_path, data=mse_summary_records)

            # Draw one pooled calibration curve per model for each fold.
            comparison_dir = f'{self.metrics_dir}/{metric_name}/By_Fold'
            dir.create_nested(comparison_dir)
            n_bins = 10
            for fold_id, predictions_by_model in pooled_predictions_by_fold.items():
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')

                for model_name in MODEL_NAMES:
                    y_pooled, p_pooled = predictions_by_model[model_name]
                    pooled_mse = mean_squared_error(y_pooled, p_pooled)
                    CalibrationDisplay.from_predictions(
                        y_pooled,
                        p_pooled,
                        n_bins=n_bins,
                        strategy='uniform',
                        ax=ax,
                        ref_line=False,
                        name=f'{model_name} (MSE={pooled_mse:.3f})',
                        marker='o',
                    )

                ax.set_xlabel('Mean Predicted Recidivism Probability')
                ax.set_ylabel('Proportion of Defendants with is_recid = 1')
                ax.set_title(
                    f'{self.instability_type} Instability - '
                    f'{eval_name} - Calibration Curve - Fold {fold_id} '
                    f'(All Runs Pooled)'
                )
                ax.text(
                    0.02, 0.98,
                    f'Bins: {n_bins} (uniform)',
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.8)
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.legend()
                fig.tight_layout()
                img_file_path = (
                    f'{comparison_dir}/{metric_name}_{self.instability_type}_'
                    f'{eval_name}_Fold_{fold_id}.png'
                )
                fig.savefig(img_file_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
            print()
    
    # Instability metric
    def ninety_five_prediction_error(self):
        print('[Metric] 95 Prediction Error')
        metric_name = '95_Prediction_Error'
        
        for eval_name in EVAL_NAMES:
            fold_model_errors = {
                fold_id: {}
                for fold_id in range(1, self.no_of_folds + 1)
            }

            for model_name in MODEL_NAMES:
                pe_dir = f'{self.metrics_dir}/{metric_name}/{model_name}'
                dir.create_nested(pe_dir)

                summary_csv_path = self.generate_csv_path(pe_dir, model_name, metric_name, eval_name)
                
                summary_dict = {
                    'Fold': [],
                    'Lower_95_Error': [],
                    'Upper_95_Error': [],
                    'Mean_Error': [],
                    'Median_Error': [],
                }
                
                for fold_id in range(1, self.no_of_folds + 1):
                    self.print_metric_progress(eval_name, model_name, fold_id, '95% Prediction Error', 100)
                    predictions = self.get_predictions(model_name, eval_name, fold_id)
                    y_eval = np.asarray(self.get_y_values(eval_name, fold_id)).ravel()

                    errors = np.abs(y_eval[:, np.newaxis] - predictions)
                    raw_csv_path = self.generate_csv_path(pe_dir, model_name, metric_name, eval_name, fold_id=fold_id)
                    csva.save_to_csv(raw_csv_path, data=errors)
                    
                    # Get mean error each run
                    mean_errors = np.mean(errors, axis=0)
                    flat_mean_errors = mean_errors.to_numpy().flatten()
                    fold_model_errors[fold_id][model_name] = flat_mean_errors
                    
                    summary_dict['Fold'].append(fold_id)
                    summary_dict['Lower_95_Error'].append(np.percentile(flat_mean_errors, 2.5))
                    summary_dict['Upper_95_Error'].append(np.percentile(flat_mean_errors, 97.5))
                    summary_dict['Mean_Error'].append(np.mean(flat_mean_errors))
                    summary_dict['Median_Error'].append(np.median(flat_mean_errors))
                csva.save_to_csv(summary_csv_path, data=summary_dict)

            # Compare each model's per-run mean absolute errors within each fold.
            comparison_dir = f'{self.metrics_dir}/{metric_name}/By_Fold'
            dir.create_nested(comparison_dir)
            for fold_id, errors_by_model in fold_model_errors.items():
                plot_path = (
                    f'{comparison_dir}/{metric_name}_{self.instability_type}_'
                    f'{eval_name}_Fold_{fold_id}_Boxplot.png'
                )
                model_errors = [errors_by_model[model_name] for model_name in MODEL_NAMES]

                plt.figure(figsize=(12, 6))
                plt.boxplot(
                    model_errors,
                    tick_labels=MODEL_NAMES,
                    whis=(2.5, 97.5),
                    notch=True
                )
                plt.xlabel('Model')
                plt.ylabel('Mean Absolute Prediction Error per Run')
                plt.title(
                    f'{self.instability_type} - 95% Prediction Error - '
                    f'{eval_name} - Fold {fold_id}'
                )
                plt.tight_layout()
                plt.autoscale(axis='y', tight=True)
                plt.savefig(plot_path)
                plt.close()
            print()
                
    # Instability metric
    def mean_absolute_prediction_error(self): 
        print('[Metric] Mean Absolute Prediction Error (MAPE)')
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
                    
                    fold_csv_path = self.generate_csv_path(mape_dir, model_name, metric_name, eval_name, fold_id=fold_id)
                    csva.save_to_csv(fold_csv_path, 'Defendant_ID', self.get_defendant_ids(eval_name, fold_id))
                    csva.save_to_csv(fold_csv_path, 'MAPE', mape_per_defendant)
                    
                    summary_dict['Mean_MAPE'].append(np.mean(mape_per_defendant))
                    print()
                    
                summary_csv_path = self.generate_csv_path(mape_dir, model_name, metric_name, eval_name)
                csva.save_to_csv(summary_csv_path, data=summary_dict)
            
    # Instability metric 
    def top_k_jaccard(self):
        print('[Metric] Top-K Jaccard')
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
                        
                    fold_csv_path = self.generate_csv_path(tk_jaccard_dir, model_name, metric_name, eval_name, fold_id=fold_id)
                    csva.save_to_csv(fold_csv_path, data=jaccard_fold_dict)
                        
                    summary_dict[f'Mean_{metric_name}'].append(np.mean(jaccard_fold_dict['Jaccard']))
                    summary_dict[f'STD_{metric_name}'].append(np.std(jaccard_fold_dict['Jaccard'], ddof=1))
                    print()
                    
                summary_csv_path = self.generate_csv_path(tk_jaccard_dir, model_name, metric_name, eval_name)
                csva.save_to_csv(summary_csv_path, data=summary_dict)
                
    # Instability metric 
    def classification_instability_index(self):
        print('[Metric] Classification Instability Index (CII)')
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
                    
                    fold_csv_path = self.generate_csv_path(cii_dir, model_name, metric_name, eval_name, fold_id=fold_id)
                    
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
                    
                summary_csv_path = self.generate_csv_path(cii_dir, model_name, metric_name, eval_name)
                csva.save_to_csv(summary_csv_path, data=summary_dict)
                            
    # Fairness metric: subgroup false positive rates and their max-min gap.
    def demographic_false_positive_rate(self):
        print('[Metric] Race-specific FPR and Maximum FPR Difference (DFPR)')
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

                    delta_csv_path = self.generate_csv_path(dfpr_dir, model_name, metric_name, eval_name, custom_title='Delta_FPR', fold_id=fold_id)
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
                    fold_csv_path = self.generate_csv_path(dfpr_dir, model_name, metric_name, eval_name, fold_id=fold_id)
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

                mean_csv_path = self.generate_csv_path(dfpr_dir, model_name, metric_name, eval_name, custom_title='Means')
                pd.DataFrame(mean_dict).to_csv(mean_csv_path, index=False)
                
                std_csv_path = self.generate_csv_path(dfpr_dir, model_name, metric_name, eval_name, custom_title='STD')
                pd.DataFrame(std_dict).to_csv(std_csv_path, index=False)
                print()
                
                
    # SHAP ANALYSIS
    # Uses the runtime shap values per fold
    def shap_analysis(self):
        print('[Metric] SHAP Analysis')
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
                    run_columns = shap_columns.drop(
                        ['Defendant_ID', 'Feature'],
                        errors='ignore'
                    )
                    explanation = shap.Explanation(
                        values=shap_values[run_columns].to_numpy(dtype=float).T,
                        feature_names=shap_values['Feature'].astype(str).tolist()
                    )

                    plt.figure(figsize=(10, 8))
                    shap.plots.beeswarm(
                        explanation,
                        max_display=15,
                        color='#1f77b4',
                        color_bar=False,
                        show=False
                    )
                    selected_id = shap_values['Defendant_ID'].iloc[0]
                    plt.title(
                        f'SHAP values across runs - {eval_name}, {model_name}, '
                        f'Fold {fold_id}, Defendant {selected_id}'
                    )
                    plt.tight_layout()

                    shap_mean = shap_values[run_columns].mean(axis=1)
                    output = {
                        'Defendant_ID': shap_values['Defendant_ID'],
                        'Feature': shap_values['Feature'],
                        'SHAP_Mean': shap_mean
                    }
                    
                    shap_csv_path = self.generate_csv_path(shap_dir, model_name, metric_name, eval_name, fold_id=fold_id)
                    plt.savefig(shap_csv_path.replace('.csv', '.png'), bbox_inches="tight")
                    plt.close()
                    csva.save_to_csv(csv_path=shap_csv_path, data=output)    
        print()
