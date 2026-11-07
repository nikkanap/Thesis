import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
from matplotlib.patches import Patch

from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    brier_score_loss, 
    mean_squared_error,
    accuracy_score, f1_score, precision_score, confusion_matrix
)

from itertools import combinations

import directory as dir
import csv_actions as csva

# Constant Variables
DFPR_THRESHOLD = 0.5
CII_THRESHOLD = 0.5
DCA_THRESHOLDS = np.linspace(0.01, 0.99, 99)

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


def _ordered_bin_summary(x_values, y_values, n_bins=10, percentiles=(25, 75)):
    """Summarize y values across equally populated, x-ordered bins."""
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]
    if not len(x_values):
        return tuple(np.array([]) for _ in range(4))

    order = np.argsort(x_values)
    bins = np.array_split(order, min(n_bins, len(order)))
    centers, medians, lowers, uppers = [], [], [], []
    for indices in bins:
        centers.append(float(np.median(x_values[indices])))
        medians.append(float(np.median(y_values[indices])))
        lower, upper = np.percentile(y_values[indices], percentiles)
        lowers.append(float(lower))
        uppers.append(float(upper))
    return tuple(np.asarray(values) for values in (centers, medians, lowers, uppers))

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

    def get_original_predictions(self, model_name, eval_name, fold_id):
        csv_path = (
            f'{self.predictions_dir}/Original_{model_name}_Predictions_'
            f'{eval_name}_{fold_id}.csv'
        )
        return csva.load_csv_data(csv_path)['Original']

    def get_reference_predictions(
        self, model_name, eval_name, fold_id, predictions=None
    ):
        """Get the comparison predictions for instability metrics."""
        if self.instability_type == 'Stochastic':
            if predictions is None:
                predictions = self.get_predictions(model_name, eval_name, fold_id)
            return predictions.mean(axis=1)
        return self.get_original_predictions(model_name, eval_name, fold_id)
    
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
                        f'{self.instability_type}_{eval_name}_Fold_{fold_id}_Mean.png'
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
    
    # Supplementary classification metrics
    def supplementary_classification_metrics(self):
        print('[Metric] Supplementary Classification Metrics')
        metric_name = 'Supplementary_Classification_Metrics'
        metric_columns = ['F1', 'Precision', 'Accuracy', 'FPR', 'TPR']

        for eval_name in EVAL_NAMES:
            for model_name in MODEL_NAMES:
                metric_dir = f'{self.metrics_dir}/{metric_name}/{model_name}'
                dir.create_nested(metric_dir)
                summary_records = []

                for fold_id in range(1, self.no_of_folds + 1):
                    predictions = self.get_predictions(model_name, eval_name, fold_id)
                    y_eval = np.asarray(self.get_y_values(eval_name, fold_id)).ravel().astype(int)
                    run_records = []

                    for i, col in enumerate(predictions.columns, start=1):
                        self.print_metric_progress(
                            eval_name,
                            model_name,
                            fold_id,
                            'Supplementary Classification Metrics',
                            i,
                            len(predictions.columns),
                        )
                        y_pred = (predictions[col].to_numpy() >= DFPR_THRESHOLD).astype(int)
                        tn, fp, fn, tp = confusion_matrix(
                            y_eval,
                            y_pred,
                            labels=[0, 1],
                        ).ravel()

                        run_records.append({
                            'Run': col,
                            'Threshold': DFPR_THRESHOLD,
                            'F1': f1_score(y_eval, y_pred, zero_division=0),
                            'Precision': precision_score(y_eval, y_pred, zero_division=0),
                            'Accuracy': accuracy_score(y_eval, y_pred),
                            'FPR': fp / (fp + tn) if (fp + tn) else np.nan,
                            'TPR': tp / (tp + fn) if (tp + fn) else np.nan,
                        })

                    run_path = self.generate_csv_path(
                        metric_dir,
                        model_name,
                        metric_name,
                        eval_name,
                        custom_title='Per_Run',
                        fold_id=fold_id,
                    )
                    run_df = pd.DataFrame(run_records)
                    run_df.to_csv(run_path, index=False)

                    summary_record = {'Fold': fold_id}
                    for metric in metric_columns:
                        values = run_df[metric].to_numpy(dtype=float)
                        valid_values = values[np.isfinite(values)]
                        summary_record[f'Mean_{metric}'] = (
                            float(np.mean(valid_values)) if len(valid_values) else np.nan
                        )
                        summary_record[f'SD_{metric}'] = (
                            float(np.std(valid_values, ddof=1))
                            if len(valid_values) > 1 else np.nan
                        )
                    summary_records.append(summary_record)

                summary_path = self.generate_csv_path(
                    metric_dir,
                    model_name,
                    metric_name,
                    eval_name,
                    custom_title='Summary',
                )
                pd.DataFrame(summary_records).to_csv(summary_path, index=False)
                print()

    # Prediction instability across defendants and model runs
    def per_defendant_prediction_variability(self):
        print('[Metric] Per-Defendant Prediction Variability')
        metric_name = 'Per_Defendant_Prediction_Variability'

        for eval_name in EVAL_NAMES:
            fold_model_instability = {
                fold_id: {}
                for fold_id in range(1, self.no_of_folds + 1)
            }

            for model_name in MODEL_NAMES:
                metric_dir = f'{self.metrics_dir}/{metric_name}/{model_name}'
                dir.create_nested(metric_dir)
                summary_records = []

                for fold_id in range(1, self.no_of_folds + 1):
                    predictions = self.get_predictions(model_name, eval_name, fold_id)
                    defendant_ids = np.asarray(self.get_defendant_ids(eval_name, fold_id)).ravel()
                    run_values = predictions.to_numpy(dtype=float)
                    
                    mean_predictions = np.mean(run_values, axis=1)
                    sd_predictions = (
                        np.std(run_values, axis=1, ddof=1)
                        if run_values.shape[1] > 1
                        else np.zeros(run_values.shape[0], dtype=float)
                    )
                    
                    lower_95, upper_95 = np.percentile(run_values, [2.5, 97.5], axis=1)
                    interval_widths = upper_95 - lower_95
                    fold_model_instability[fold_id][model_name] = sd_predictions

                    raw_records = predictions.copy()
                    raw_records.insert(0, 'Defendant_ID', defendant_ids)
                    raw_records['Mean_Prediction'] = mean_predictions
                    raw_records['SD_Prediction'] = sd_predictions
                    raw_records['Lower_95'] = lower_95
                    raw_records['Upper_95'] = upper_95
                    raw_records['Prediction_Interval_Width'] = interval_widths
                    raw_path = self.generate_csv_path(metric_dir, model_name, metric_name, eval_name, custom_title='Raw', fold_id=fold_id)
                    raw_records.to_csv(raw_path, index=False)

                    summary_records.append({
                        'Fold': fold_id,
                        'Defendant_Count': len(defendant_ids),
                        'Run_Count': run_values.shape[1],
                        'Mean_Defendant_SD': float(np.mean(sd_predictions)),
                        'Median_Defendant_SD': float(np.median(sd_predictions)),
                        'P2_5_Defendant_SD': float(np.percentile(sd_predictions, 2.5)),
                        'P97_5_Defendant_SD': float(np.percentile(sd_predictions, 97.5)),
                        'Mean_95PI_Width': float(np.mean(interval_widths)),
                        'Median_95PI_Width': float(np.median(interval_widths)),
                    })

                summary_path = self.generate_csv_path(
                    metric_dir,
                    model_name,
                    metric_name,
                    eval_name,
                    custom_title='Summary',
                )
                pd.DataFrame(summary_records).to_csv(summary_path, index=False)

            # Compare all models' per-defendant instability across folds.
            plot_dir = f'{self.metrics_dir}/{metric_name}/By_Evaluation'
            dir.create_nested(plot_dir)
            colors = plt.get_cmap('tab10')
            fold_ids = list(fold_model_instability)
            group_width = 0.8
            model_spacing = group_width / len(MODEL_NAMES)
            model_offsets = (
                np.arange(len(MODEL_NAMES)) - (len(MODEL_NAMES) - 1) / 2
            ) * model_spacing
            box_positions = []
            model_predictions = []
            fold_tick_labels = []

            for fold_id in fold_ids:
                fold_tick_labels.append(f'Fold {fold_id}')
                for model_index, model_name in enumerate(MODEL_NAMES):
                    model_predictions.append(fold_model_instability[fold_id][model_name])
                    box_positions.append(fold_id + model_offsets[model_index])

            fig, ax = plt.subplots(figsize=(max(10, 2.4 * len(fold_ids)), 6))
            boxplot = ax.boxplot(
                model_predictions,
                positions=box_positions,
                widths=model_spacing * 0.8,
                whis=(2.5, 97.5),
                patch_artist=True,
                showfliers=False,
            )
            for i, box in enumerate(boxplot['boxes']):
                box.set_facecolor(colors(i % len(MODEL_NAMES)))
                box.set_alpha(0.45)

            ax.set_xticks(fold_ids, fold_tick_labels)
            ax.set_xlabel('Fold')
            ax.set_ylabel('Per-Defendant SD Across Runs')
            ax.set_title(
                f'{self.instability_type} - {eval_name} - '
                'Prediction Instability Across Defendants and Folds'
            )
            ax.text(
                0.02,
                0.98,
                'Each box summarizes defendants’ SD across runs',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8),
            )
            ax.legend(
                handles=[
                    Patch(facecolor=colors(i), alpha=0.45, label=model_name)
                    for i, model_name in enumerate(MODEL_NAMES)
                ],
                title='Model',
            )
            ax.autoscale(axis='y', tight=True)
            fig.tight_layout()
            plot_path = (
                f'{plot_dir}/{metric_name}_{self.instability_type}_'
                f'{eval_name}_Across_Folds.png'
            )
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
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

                    fig, ax = plt.subplots(figsize=(6, 6))
                    boxplot = ax.boxplot(
                        [flat_mean_errors],
                        tick_labels=[model_name],
                        whis=(2.5, 97.5),
                        notch=True,
                        patch_artist=True,
                        showfliers=False,
                    )
                    boxplot['boxes'][0].set_facecolor(
                        plt.get_cmap('tab10')(MODEL_NAMES.index(model_name))
                    )
                    boxplot['boxes'][0].set_alpha(0.45)
                    ax.set_xlabel('Model')
                    ax.set_ylabel('Mean Absolute Prediction Error per Run')
                    ax.set_title(
                        f'{self.instability_type} - {model_name} - 95% '
                        f'Prediction Error - {eval_name} - Fold {fold_id}'
                    )
                    ax.autoscale(axis='y', tight=True)
                    fig.tight_layout()
                    individual_plot_path = (
                        f'{pe_dir}/{metric_name}_{model_name}_'
                        f'{self.instability_type}_{eval_name}_Fold_{fold_id}_Boxplot.png'
                    )
                    fig.savefig(individual_plot_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                csva.save_to_csv(summary_csv_path, data=summary_dict)

            # Compare per-run mean absolute errors across models and folds.
            comparison_dir = f'{self.metrics_dir}/{metric_name}/By_Evaluation'
            dir.create_nested(comparison_dir)
            fold_ids = list(fold_model_errors)
            group_width = 0.8
            model_spacing = group_width / len(MODEL_NAMES)
            model_offsets = (
                np.arange(len(MODEL_NAMES)) - (len(MODEL_NAMES) - 1) / 2
            ) * model_spacing
            box_positions = []
            model_errors = []

            for fold_id in fold_ids:
                errors_by_model = fold_model_errors[fold_id]
                for model_index, model_name in enumerate(MODEL_NAMES):
                    model_errors.append(errors_by_model[model_name])
                    box_positions.append(fold_id + model_offsets[model_index])

            fig, ax = plt.subplots(figsize=(max(10, 2.4 * len(fold_ids)), 6))
            boxplot = ax.boxplot(
                model_errors,
                positions=box_positions,
                widths=model_spacing * 0.8,
                whis=(2.5, 97.5),
                patch_artist=True,
                showfliers=False,
            )
            colors = plt.get_cmap('tab10')
            for i, box in enumerate(boxplot['boxes']):
                box.set_facecolor(colors(i % len(MODEL_NAMES)))
                box.set_alpha(0.45)

            ax.set_xticks(fold_ids, [f'Fold {fold_id}' for fold_id in fold_ids])
            ax.set_xlabel('Fold')
            ax.set_ylabel('Mean Absolute Prediction Error per Run')
            ax.set_title(
                f'{self.instability_type} - 95% Prediction Error '
                f'Across Folds - {eval_name}'
            )
            ax.legend(
                handles=[
                    Patch(facecolor=colors(i), alpha=0.45, label=model_name)
                    for i, model_name in enumerate(MODEL_NAMES)
                ],
                title='Model',
            )
            ax.autoscale(axis='y', tight=True)
            fig.tight_layout()
            plot_path = (
                f'{comparison_dir}/{metric_name}_{self.instability_type}_'
                f'{eval_name}_Across_Folds_Boxplot.png'
            )
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print()
                
    # Instability metric: run predictions relative to a reference prediction.
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
                    reference_predictions = self.get_reference_predictions(
                        model_name, eval_name, fold_id, predictions
                    ).to_numpy(dtype=float)
                    mape_per_defendant = np.abs(
                        predictions.to_numpy(dtype=float)
                        - reference_predictions[:, np.newaxis]
                    ).mean(axis=1)

                    fold_csv_path = self.generate_csv_path(mape_dir, model_name, metric_name, eval_name, fold_id=fold_id)
                    pd.DataFrame({
                        'Defendant_ID': self.get_defendant_ids(eval_name, fold_id),
                        'Reference_Prediction': reference_predictions,
                        'MAPE': mape_per_defendant,
                    }).to_csv(fold_csv_path, index=False)
                    
                    summary_dict['Mean_MAPE'].append(np.mean(mape_per_defendant))
                    print()
                    
                summary_csv_path = self.generate_csv_path(mape_dir, model_name, metric_name, eval_name)
                csva.save_to_csv(summary_csv_path, data=summary_dict)

    def mape_instability_plot(self):
        """Plot each defendant's MAPE against their reference prediction."""
        print('[Plot] MAPE Instability Plot')
        metric_name = 'MAPE'

        for eval_name in EVAL_NAMES:
            for model_name in MODEL_NAMES:
                mape_dir = f'{self.metrics_dir}/{metric_name}/{model_name}'
                for fold_id in range(1, self.no_of_folds + 1):
                    csv_path = self.generate_csv_path(
                        mape_dir,
                        model_name,
                        metric_name,
                        eval_name,
                        fold_id=fold_id,
                    )
                    mape_data = pd.read_csv(csv_path)
                    x_values = mape_data['Reference_Prediction'].to_numpy(dtype=float)
                    y_values = mape_data['MAPE'].to_numpy(dtype=float)
                    bin_x, bin_median, bin_lower, bin_upper = _ordered_bin_summary(
                        x_values, y_values, percentiles=(25, 75)
                    )

                    fig, ax = plt.subplots(figsize=(7, 6))
                    ax.scatter(
                        x_values,
                        y_values,
                        s=18,
                        alpha=0.35,
                        edgecolors='none',
                        label='Defendants',
                    )
                    if len(bin_x):
                        ax.fill_between(
                            bin_x,
                            bin_lower,
                            bin_upper,
                            color='darkorange',
                            alpha=0.32,
                            label='25th–75th percentile by risk bin',
                        )
                        ax.plot(
                            bin_x,
                            bin_median,
                            color='darkorange',
                            linewidth=2,
                            marker='o',
                            markersize=4,
                            label='Median MAPE by risk bin',
                        )
                    x_range = float(np.ptp(x_values)) if len(x_values) else 0.0
                    x_padding = max(0.01, x_range * 0.04)
                    x_min = max(0.0, float(np.min(x_values)) - x_padding)
                    x_max = min(1.0, float(np.max(x_values)) + x_padding)
                    y_max = min(1.0, max(0.02, float(np.max(y_values)) * 1.12))
                    ax.set_xlim(x_min, x_max if x_max > x_min else min(1.0, x_min + 0.02))
                    ax.set_ylim(0, y_max)
                    ax.set_xlabel('Reference Prediction')
                    ax.set_ylabel('Per-Defendant Mean Absolute Prediction Error')
                    ax.set_title(
                        f'{self.instability_type} - {model_name} - {eval_name} '
                        f'- Fold {fold_id} - MAPE Instability'
                    )
                    ax.text(
                        0.98,
                        0.97,
                        f'Defendants: {len(y_values):,}\n'
                        f'Mean MAPE: {np.mean(y_values):.4f}\n'
                        f'Median MAPE: {np.median(y_values):.4f}\n'
                        f'95th percentile: {np.percentile(y_values, 95):.4f}',
                        transform=ax.transAxes,
                        ha='right',
                        va='top',
                        bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.9),
                    )
                    ax.legend(loc='best', fontsize=8)
                    ax.grid(alpha=0.2)
                    fig.tight_layout()

                    plot_path = (
                        f'{mape_dir}/MAPE_Instability_{self.instability_type}_'
                        f'{eval_name}_Fold_{fold_id}.png'
                    )
                    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)

    # Compare each run's prediction with the reference for each evaluation individual.
    def prediction_instability_scatter(self):
        print('[Metric] Prediction Instability Scatter')
        metric_dir = f'{self.metrics_dir}/Prediction_Instability_Scatter'
        dir.create_nested(metric_dir)

        for eval_name in EVAL_NAMES:
            for model_name in MODEL_NAMES:
                model_dir = f'{metric_dir}/{model_name}'
                dir.create_nested(model_dir)
                for fold_id in range(1, self.no_of_folds + 1):
                    predictions = self.get_predictions(
                        model_name, eval_name, fold_id
                    )
                    reference = self.get_reference_predictions(
                        model_name, eval_name, fold_id, predictions
                    ).to_numpy(dtype=float)
                    values = predictions.to_numpy(dtype=float)
                    defendant_ids = np.asarray(
                        self.get_defendant_ids(eval_name, fold_id)
                    ).ravel()

                    raw_records = []
                    for run_index, run_name in enumerate(predictions.columns):
                        raw_records.append(pd.DataFrame({
                            'Defendant_ID': defendant_ids,
                            'Run': run_name,
                            'Reference_Prediction': reference,
                            'Run_Prediction': values[:, run_index],
                        }))
                    raw_df = pd.concat(raw_records, ignore_index=True)
                    raw_df['Absolute_Difference'] = np.abs(
                        raw_df['Run_Prediction']
                        - raw_df['Reference_Prediction']
                    )
                    raw_path = (
                        f'{model_dir}/Prediction_Instability_Scatter_'
                        f'{self.instability_type}_{eval_name}_Fold_{fold_id}_Raw.csv'
                    )
                    raw_df.to_csv(raw_path, index=False)

                    bin_x, bin_median, bin_lower, bin_upper = _ordered_bin_summary(
                        raw_df['Reference_Prediction'],
                        raw_df['Run_Prediction'],
                        percentiles=(25, 75),
                    )
                    per_defendant_mape = raw_df.groupby(
                        'Defendant_ID'
                    )['Absolute_Difference'].mean()
                    per_defendant_sd = raw_df.groupby(
                        'Defendant_ID'
                    )['Run_Prediction'].std(ddof=1)

                    fig, ax = plt.subplots(figsize=(7, 7))
                    ax.scatter(
                        raw_df['Reference_Prediction'],
                        raw_df['Run_Prediction'],
                        s=9,
                        alpha=0.12,
                        edgecolors='none',
                        label='Run predictions',
                    )
                    if len(bin_x):
                        ax.fill_between(
                            bin_x,
                            bin_lower,
                            bin_upper,
                            color='darkorange',
                            alpha=0.32,
                            label='25th–75th percentile by risk bin',
                        )
                        ax.plot(
                            bin_x,
                            bin_median,
                            color='darkorange',
                            linewidth=2,
                            marker='o',
                            markersize=4,
                            label='Median run prediction by risk bin',
                        )
                    values_for_limits = np.concatenate([
                        raw_df['Reference_Prediction'].to_numpy(dtype=float),
                        raw_df['Run_Prediction'].to_numpy(dtype=float),
                    ])
                    data_range = float(np.ptp(values_for_limits))
                    padding = max(0.01, data_range * 0.04)
                    axis_min = max(0.0, float(np.min(values_for_limits)) - padding)
                    axis_max = min(1.0, float(np.max(values_for_limits)) + padding)
                    ax.plot(
                        [axis_min, axis_max],
                        [axis_min, axis_max],
                        linestyle='--',
                        color='black',
                        linewidth=1,
                        label='Agreement line',
                    )
                    ax.set_xlim(axis_min, axis_max)
                    ax.set_ylim(axis_min, axis_max)
                    ax.set_aspect('equal', adjustable='box')
                    ax.set_xlabel('Reference Prediction')
                    ax.set_ylabel('Bootstrap Model Prediction')
                    ax.set_title(
                        f'{self.instability_type} - {model_name} - '
                        f'{eval_name} - Fold {fold_id} - Prediction Instability'
                    )
                    ax.text(
                        0.02,
                        0.98,
                        f'Defendants: {len(defendant_ids):,}\n'
                        f'Bootstrap fits: {predictions.shape[1]:,}\n'
                        f'Mean MAPE: {per_defendant_mape.mean():.4f}\n'
                        f'Mean defendant SD: {per_defendant_sd.mean():.4f}',
                        transform=ax.transAxes,
                        ha='left',
                        va='top',
                        bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.9),
                    )
                    ax.legend(loc='lower right', fontsize=8)
                    ax.grid(alpha=0.2)
                    fig.tight_layout()
                    plot_path = (
                        f'{model_dir}/Prediction_Instability_Scatter_'
                        f'{self.instability_type}_{eval_name}_Fold_{fold_id}.png'
                    )
                    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
            
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
        threshold = CII_THRESHOLD
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
                    reference_predictions = self.get_reference_predictions(
                        model_name, eval_name, fold_id, predictions
                    ).to_numpy(dtype=float)
                    reference_labels = reference_predictions >= threshold
                    bootstrap_labels = predictions.to_numpy(dtype=float) >= threshold
                    cii_individual = np.mean(
                        bootstrap_labels != reference_labels[:, np.newaxis],
                        axis=1,
                    )

                    fold_csv_path = self.generate_csv_path(cii_dir, model_name, metric_name, eval_name, fold_id=fold_id)
                    defendant_ids = np.asarray(
                        self.get_defendant_ids(eval_name, fold_id)
                    ).ravel()
                    pd.DataFrame({
                        'Defendant_ID': defendant_ids,
                        'Reference_Prediction': reference_predictions,
                        'Reference_Class': reference_labels.astype(int),
                        'CII': cii_individual,
                        'Threshold': threshold,
                    }).to_csv(fold_csv_path, index=False)

                    summary_dict[f'Mean_{metric_name}'].append(np.mean(cii_individual))
                    summary_dict[f'STD_{metric_name}'].append(np.std(cii_individual, ddof=1))

                    fig, ax = plt.subplots(figsize=(7, 6))
                    ax.scatter(
                        reference_predictions,
                        cii_individual,
                        s=18,
                        alpha=0.45,
                        edgecolors='none',
                    )
                    ax.axvline(
                        threshold,
                        color='black',
                        linestyle='--',
                        linewidth=1,
                        label=f'Classification threshold ({threshold:.2f})',
                    )
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_xlabel('Reference Prediction')
                    ax.set_ylabel('Per-Defendant Classification Instability Index')
                    ax.set_title(
                        f'{self.instability_type} - {model_name} - {eval_name} '
                        f'- Fold {fold_id} - CII'
                    )
                    ax.text(
                        0.98,
                        0.97,
                        f'Defendants: {len(cii_individual):,}\n'
                        f'Bootstrap fits: {predictions.shape[1]:,}\n'
                        f'Mean CII: {np.mean(cii_individual):.4f}',
                        transform=ax.transAxes,
                        ha='right',
                        va='top',
                        bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.9),
                    )
                    ax.legend(loc='best', fontsize=8)
                    ax.grid(alpha=0.2)
                    fig.tight_layout()
                    fig.savefig(fold_csv_path.replace('.csv', '.png'), dpi=300, bbox_inches='tight')
                    plt.close(fig)
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
