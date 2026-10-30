import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, r2_score

from itertools import combinations
from statsmodels.nonparametric.smoothers_lowess import lowess

from create_dir import create_nested_directory
import logger

RACE_NAMES = [
    'race:African-American',
    'race:Asian',
    'race:Caucasian',
    'race:Native_American',
    'race:Hispanic',
    'race:Other'
]
DFPR_THRESHOLD = 0.5
MODEL_NAMES = [ 'LR', 'RF', 'LSVC', 'XGB', 'MLP' ]
TEST_NAMES = [ 'Validation', 'Test'] 

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
            logger.exception(f"An error occurred: ", e)
    
    # Saves data to csv files
    def save_to_csv(
        self, 
        csv_file_path,
        metric_column_name,
        metric_column_data
    ):
        if os.path.isfile(csv_file_path):
            df = pd.read_csv(csv_file_path)
            df[metric_column_name] = metric_column_data
        else:
            df = pd.DataFrame({
                metric_column_name: metric_column_data
            })
        df.to_csv(csv_file_path, index=False) 
    
    # ================
    # GETTER FUNCTIONS
    # ================
    
    # Gets the defendant ids from the test or validation data
    def get_defendant_ids(self, test_name, fold_id=None):
        return (
            self.test_defendant_ids
            if test_name == 'Test'
            else self.val_defendant_ids_list[fold_id-1]
        )
    
    # Gets the y_values from the test or validation data
    def get_y_values(self, test_name, fold_id=None):        
        return (
            self.y_test 
            if test_name == 'Test' 
            else self.y_val_list[fold_id-1]
        )
                     
    # Gets the predictions per fold
    def get_predictions(self, model_name, test_name, fold_id):
        return pd.read_csv(f'{self.predictions_dir}/{model_name}_Predictions_{test_name}_{fold_id}.csv')                           
    
    
    
    # =======
    # METRICS
    # =======
    # Performance metric
    def roc_auc(self):
        print(f'Metric: ROC-AUC')
        metric_name = 'ROC_AUC'
        
        for test_name in TEST_NAMES:
            for model_name in MODEL_NAMES:
                roc_auc_dir = f'{self.metrics_dir}/roc_auc/{model_name}'
                create_nested_directory(roc_auc_dir)
                
                roc_auc_means = []
                roc_auc_stds = []
                
                for fold_id in range(1, self.no_of_folds + 1):
                    # Gets the predictions
                    predictions = self.get_predictions(model_name, test_name, fold_id)
                    
                    # Saves the 
                    fold_csv_file_path = f'{roc_auc_dir}/{metric_name}_{model_name}_{self.instability_type}_{test_name}_Fold_{fold_id}.csv'
                    self.save_to_csv(
                        fold_csv_file_path,
                        'Run',
                        predictions.columns
                    )
                    
                    # Getting the y_values of test/validation
                    y_eval = self.get_y_values(test_name, fold_id)
                    
                    roc_aucs = []
                    for col in predictions.columns:
                        print(f'{test_name}, {model_name}, Fold {fold_id}/{self.no_of_folds} - Getting ROC-AUCs... {round((len(roc_aucs)/len(predictions.columns))*100)}%', end='\r')
                        
                        # Get the predictions per bootstrap
                        y_pred_proba = predictions[col].values
                        
                        # Get the ROC-AUC scores for that specific run
                        roc_auc = roc_auc_score(y_eval, y_pred_proba)
                        roc_aucs.append(roc_auc)
                        
                    print(f'{test_name}, {model_name}, Fold {fold_id}/{self.no_of_folds} - Getting ROC-AUCs... 100%')
                    self.save_to_csv(
                        fold_csv_file_path,
                        'ROC_AUC_Score',
                        roc_aucs
                    )
                    
                    # Get the mean ROC-AUC Values
                    roc_auc_mean = np.mean(roc_aucs)
                    roc_auc_means.append(roc_auc_mean)
                                       
                    # Get the mean  
                    roc_auc_std = np.std(roc_aucs, ddof=1)
                    roc_auc_stds.append(roc_auc_std)
                
                agg_csv_file_path = f'{roc_auc_dir}/Aggregated_{metric_name}_{model_name}_{self.instability_type}.csv'
                if test_name == 'Validation':
                    self.save_to_csv(
                        agg_csv_file_path,
                        'Fold',
                        [i for i in range(1, self.no_of_folds + 1)]                   
                    )
                
                self.save_to_csv(
                    agg_csv_file_path,
                    f'Mean_{metric_name}',
                    roc_auc_means
                )
                
                self.save_to_csv(
                    agg_csv_file_path,
                    f'STD_{metric_name}',
                    roc_auc_stds
                )
            print()
        
    # Performance metric
    def brier_score(self):
        metric_name = 'Brier_Score'
        
        print('Metric: Brier Score')
        for test_name in TEST_NAMES:            
            for model_name in MODEL_NAMES:         
                brier_score_dir = f'{self.metrics_dir}/brier_score/{model_name}'
                create_nested_directory(brier_score_dir)
                
                brier_score_means = []
                brier_score_stds = []
                
                for fold_id in range(1, self.no_of_folds + 1):
                    fold_csv_file_path = f'{brier_score_dir}/{metric_name}_{model_name}_{self.instability_type}_{test_name}_Fold_{fold_id}.csv'       
                    y_eval = self.get_y_values(test_name, fold_id)
                                                                   
                    predictions = self.get_predictions(
                        model_name,
                        test_name,
                        fold_id=fold_id
                    )
                    
                    self.save_to_csv(
                        fold_csv_file_path,
                        'run',
                        predictions.columns
                    )
                    
                    brier_scores = []
                    for col in predictions.columns:
                        print(f'{test_name}, {model_name}, Fold {fold_id}/{self.no_of_folds} - Getting Brier Scores... {round((len(brier_scores)/len(predictions.columns))*100)}%', end='\r')
                        y_pred_proba = predictions[col].values
                        
                        brier_score = brier_score_loss(y_eval, y_pred_proba)
                        brier_scores.append(brier_score)
                    print(f'{test_name}, {model_name}, Fold {fold_id}/{self.no_of_folds} - Getting Brier Scores... 100%')
                    self.save_to_csv(
                        fold_csv_file_path,
                        'brier_score',
                        brier_scores
                    )
                    
                    # Compute mean and STD of Brier scores across runs
                    brier_score_mean = np.mean(brier_scores)
                    brier_score_means.append(brier_score_mean)
                
                    brier_score_std = np.std(brier_scores, ddof=1)
                    brier_score_stds.append(brier_score_std)
                
                agg_csv_file_path = f'{brier_score_dir}/Aggregated_{metric_name}_{model_name}_{self.instability_type}.csv'     
                if test_name == 'Validation':
                    self.save_to_csv(
                        agg_csv_file_path,
                        'fold',
                        [i for i in range(1, self.no_of_folds + 1)]                   
                    )
                
                self.save_to_csv(
                    agg_csv_file_path,
                    f'Mean_{metric_name}_{test_name}',
                    brier_score_means                   
                )    
                    
                self.save_to_csv(
                    agg_csv_file_path,
                    f'STD_{metric_name}_{test_name}',
                    brier_score_stds
                )
            print()

    # Instability metric
    def ninety_five_prediction_interval(self):
        metric_name = '95_Prediction_Interval'
        print('Metric: 95 Prediction Interval')

        for test_name in TEST_NAMES:
            for model_name in MODEL_NAMES:
                si_dir = f'{self.metrics_dir}/95_prediction_interval/{model_name}'
                create_nested_directory(si_dir)

                agg_csv_file_path = (
                    f'{si_dir}/Aggregated_{metric_name}_'
                    f'{model_name}_{self.instability_type}_{test_name}.csv'
                )

                folds = []
                lower_95_errors = []
                upper_95_errors = []
                mean_errors = []
                median_errors = []

                # Store errors for the final box plot
                all_fold_errors = []

                for fold_id in range(1, self.no_of_folds + 1):
                    print(
                        f'{test_name}, {model_name}, '
                        f'Fold {fold_id}/{self.no_of_folds}',
                        end='\r'
                    )

                    # Predictions:
                    # rows    = defendants
                    # columns = bootstrap runs
                    predictions = self.get_predictions(
                        model_name,
                        test_name,
                        fold_id
                    )

                    # Actual y values
                    y_true = self.get_y_values(
                        test_name,
                        fold_id
                    )

                    errors = np.abs(y_true[:, np.newaxis] - predictions)

                    # Flatten all defendant-bootstrap errors for this fold
                    fold_errors = errors.to_numpy().flatten()

                    # Store for final box plot
                    all_fold_errors.append(fold_errors)

                    lower_95 = np.percentile(fold_errors, 2.5)
                    upper_95 = np.percentile(fold_errors, 97.5)

                    mean_error = np.mean(fold_errors)
                    median_error = np.median(fold_errors)

                    folds.append(fold_id)
                    lower_95_errors.append(lower_95)
                    upper_95_errors.append(upper_95)
                    mean_errors.append(mean_error)
                    median_errors.append(median_error)

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

                plt.title(
                    f'{self.instability_type} - {model_name} - '
                    f'95% Prediction Error Distribution - {test_name}'
                )

                plt.tight_layout()

                plot_path = agg_csv_file_path.replace(
                    '.csv',
                    '_Prediction_Error_Boxplot.png'
                )

                plt.savefig(plot_path)
                plt.close()

                self.save_to_csv(
                    agg_csv_file_path,
                    'Fold',
                    folds
                )

                self.save_to_csv(
                    agg_csv_file_path,
                    'Lower_95_Error',
                    lower_95_errors
                )

                self.save_to_csv(
                    agg_csv_file_path,
                    'Upper_95_Error',
                    upper_95_errors
                )

                self.save_to_csv(
                    agg_csv_file_path,
                    'Mean_Error',
                    mean_errors
                )

                self.save_to_csv(
                    agg_csv_file_path,
                    'Median_Error',
                    median_errors
                )

                print(
                    f'{test_name}, {model_name}, '
                    f'Fold {self.no_of_folds}/{self.no_of_folds}'
                )
            print()
        
    # Instability metric
    def mean_absolute_prediction_error(self): 
        metric_name = 'MAPE'
        
        print('Metric: Mean Absolute Prediction Error (MAPE)')
            
        for test_name in TEST_NAMES:
            for model_name in MODEL_NAMES:
                mape_dir = f'{self.metrics_dir}/mape/{model_name}'
                
                create_nested_directory(mape_dir)
            
                mean_mapes = []
                for fold_id in range(1, self.no_of_folds + 1):
                    print(f'{test_name}, {model_name}, Fold {fold_id}/{self.no_of_folds}', end='\r')
                    fold_csv_file_path = f'{mape_dir}/{metric_name}_{model_name}_{self.instability_type}_{test_name}_Fold_{fold_id}.csv'
                            
                    predictions = self.get_predictions(
                        model_name, 
                        test_name, 
                        fold_id=fold_id
                    ).to_numpy()  
                    
                    self.save_to_csv(
                        fold_csv_file_path,
                        'defendant_id',
                        self.get_defendant_ids(test_name, fold_id)
                    )
                        
                    mean_prediction = np.mean(predictions, axis=1) 
                    mape_per_defendant = np.mean(
                        np.abs(predictions - mean_prediction[:, None]),
                        axis=1
                    )
                    
                    self.save_to_csv(
                        fold_csv_file_path,
                        'MAPE',
                        mape_per_defendant
                    )
                    
                    mape_mean = np.mean(mape_per_defendant)
                    mean_mapes.append(mape_mean)
                print(f'{test_name}, {model_name}, Fold {self.no_of_folds}/{self.no_of_folds}')
                
                agg_csv_file_path = f'{mape_dir}/Mean_{metric_name}_{model_name}_{self.instability_type}_{test_name}.csv'
                # Save the fold numbers
                self.save_to_csv(
                    agg_csv_file_path,
                    'folds',
                    [f'fold_{i}' for i in range(1, self.no_of_folds + 1)]
                )
                
                self.save_to_csv(
                    agg_csv_file_path,
                    'Mean_MAPE',
                    mean_mapes,
                )
            print()
            
    # Instability metric 
    def top_k_jaccard(self):
        metric_name = 'Top_K_Jaccard'
        
        for test_name in TEST_NAMES:
            print(f'[Generating {self.instability_type}-{test_name} Top-K Jaccard]')
            
            for model_name in MODEL_NAMES:
                tk_jaccard_dir = f'{self.metrics_dir}/top_k_jaccard/{model_name}'
                create_nested_directory(tk_jaccard_dir)
        
                jaccard_scores_mean = []
                jaccard_scores_std = []
                
                for fold_id in range(1, self.no_of_folds + 1):                    
                    fold_csv_file_path = f'{tk_jaccard_dir}/{metric_name}_{model_name}_{self.instability_type}_{test_name}_Fold_{fold_id}.csv'     
                    run_1_list = []
                    run_2_list = []
                    jaccard_scores_list = []
                    
                    predictions = self.get_predictions(
                        model_name, 
                        test_name, 
                        fold_id=fold_id
                    )
                        
                    k = int(0.10 * len(predictions))
                    for col1, col2 in combinations(predictions.columns, 2):
                        run_1_list.append(col1)
                        run_2_list.append(col2)
                        
                        run1 = predictions[col1]
                        run2 = predictions[col2]
                        
                        top_k_1 = run1.nlargest(k).index
                        top_k_2 = run2.nlargest(k).index
                        
                        intersection = len(set(top_k_1) & set(top_k_2))
                        union = len(set(top_k_1) | set(top_k_2))

                        jaccard = intersection / union
                        jaccard_scores_list.append(jaccard)
                                    
                    self.save_to_csv(
                        fold_csv_file_path,
                       'run_1',
                       run_1_list
                    )
                    self.save_to_csv(
                        fold_csv_file_path,
                        'run_2',
                        run_2_list
                    )
                    self.save_to_csv(
                        fold_csv_file_path,
                        'jaccard',
                        jaccard_scores_list
                    )
                        
                    mean_jaccard = np.mean(jaccard_scores_list)
                    jaccard_scores_mean.append(mean_jaccard)
                    
                    std_jaccard = np.std(jaccard_scores_list, ddof=1)
                    jaccard_scores_std.append(std_jaccard)
                    
                csv_file_path = f'{tk_jaccard_dir}/Aggregated_{metric_name}_{model_name}_{self.instability_type}.csv'
                if test_name == 'Validation':
                    self.save_to_csv(
                        csv_file_path,
                        'fold',
                        [i for i in range(1, self.no_of_folds + 1)]
                    )
                    
                self.save_to_csv(
                    csv_file_path,
                    f'Mean_{metric_name}_{test_name}',
                    jaccard_scores_mean
                )
                self.save_to_csv(
                    csv_file_path,
                    f'STD_{metric_name}_{test_name}',
                    jaccard_scores_std
                )
            print()
            
    # Instability metric 
    def classification_instability_index(self):
        threshold = 0.5
        metric_name = 'CII'
        
        print('Metric: Classification Instability Index (CII)')
        for test_name in TEST_NAMES:
            for model_name in MODEL_NAMES:
                cii_dir = f'{self.metrics_dir}/cii/{model_name}'
                create_nested_directory(cii_dir)
        
                mean_ciis = []
                std_ciis = []
                for fold_id in range(1, self.no_of_folds + 1):
                    print(f'{test_name}, {model_name}, Fold {fold_id}/{self.no_of_folds}', end='\r')
                    fold_csv_file_path = f'{cii_dir}/{metric_name}_{model_name}_{self.instability_type}_{test_name}_Fold_{fold_id}.csv'    
                    predictions = self.get_predictions(
                        model_name, 
                        test_name, 
                        fold_id=fold_id
                    )
                    
                    defendant_ids = self.get_defendant_ids(test_name, fold_id)
                                    
                    self.save_to_csv(
                        fold_csv_file_path,
                        'defendant_id',
                        defendant_ids
                    )
                    
                    labels = (predictions >= threshold).astype(int)
                    
                    n_1 = labels.sum(axis=1)
                    n_0 = labels.shape[1] - n_1
                    
                    flips = n_0 * n_1
                    
                    cii_individual = flips / (labels.shape[1] * (labels.shape[1] - 1) / 2)
                    self.save_to_csv(
                        fold_csv_file_path,
                        'CII',
                        cii_individual
                    )
                                  
                    cii_mean = np.mean(cii_individual)
                    mean_ciis.append(cii_mean)
                    
                    cii_std = np.std(cii_individual, ddof=1)
                    std_ciis.append(cii_std)
                    
                    plt.hist(cii_individual, bins=20)
                    plt.xlabel('CII per Individual')
                    plt.ylabel('Count')
                    plt.title(f'Classification Instability Distribution ({self.instability_type} - {model_name}, Fold {fold_id})')
                    plt.savefig(fold_csv_file_path.replace('csv', 'png'))
                    plt.close()
                    
                print(f'{test_name}, {model_name}, Fold {self.no_of_folds}/{self.no_of_folds}')
                csv_file_path = f'{cii_dir}/Aggregated_{metric_name}_{model_name}_{self.instability_type}.csv' 
                if test_name == 'Validation':
                    self.save_to_csv(
                        csv_file_path,
                        'fold',
                        [i for i in range (1, self.no_of_folds + 1)]
                    )
                
                self.save_to_csv(
                    csv_file_path,
                    f'Mean_{metric_name}_{test_name}',
                    mean_ciis
                )
                
                self.save_to_csv(
                    csv_file_path,
                    f'STD_{metric_name}_{test_name}',
                    std_ciis
                )
            print()

    # Performance metric 
    def calibration_plot(self):
        print('Metric: Calibration Plot')
        
        metric_name = 'Calibration_Plot'
        for test_name in TEST_NAMES:
            for model_name in MODEL_NAMES:
                cal_plot_dir = f'{self.metrics_dir}/calibration_plot/{model_name}'
                create_nested_directory(cal_plot_dir)

                r2_summary_records = []
                for fold_id in range(1, self.no_of_folds + 1):
                    y_true = np.asarray(
                        self.get_y_values(test_name, fold_id)
                    ).ravel()

                    img_file_path = (
                        f'{cal_plot_dir}/{metric_name}_{model_name}_'
                        f'{self.instability_type}_{test_name}_Fold_{fold_id}.png'
                    )
                    predictions = self.get_predictions(
                        model_name,
                        test_name, 
                        fold_id=fold_id
                    )

                    plt.figure(figsize=(8, 6))
                    r2_scores = []
                    run_names = list(predictions.columns)
                    for run_index, run_name in enumerate(run_names, start=1):
                        print(
                            f'{test_name}, {model_name}, Fold '
                            f'{fold_id}/{self.no_of_folds} - Getting Calibration Plots... '
                            f'{round((run_index / len(run_names)) * 100)}%',
                            end='\r'
                        )
                        y_pred = predictions[run_name].to_numpy()

                        prob_true, prob_pred = calibration_curve(
                            y_true,
                            y_pred,
                            n_bins=10,
                            strategy='uniform'
                        )

                        plt.plot(
                            prob_pred,
                            prob_true,
                            alpha=0.1
                        )

                        # R2 is undefined when the observed outcome has no
                        # variation in this fold.
                        run_r2 = (
                            r2_score(y_true, y_pred)
                            if np.unique(y_true).size > 1
                            else np.nan
                        )
                        r2_scores.append(run_r2)

                    print(
                        f'{test_name}, {model_name}, Fold '
                        f'{fold_id}/{self.no_of_folds} - '
                        'Getting Calibration Plots... 100%'
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
                        f'{self.instability_type} - {model_name} - {test_name} - '
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

                    plt.savefig(
                        img_file_path,
                        dpi=300,
                        bbox_inches='tight'
                    )
                    plt.close()

                    per_run_r2_path = (
                        f'{cal_plot_dir}/R2_{metric_name}_{model_name}_'
                        f'{self.instability_type}_{test_name}_Fold_{fold_id}.csv'
                    )
                    pd.DataFrame({
                        'Run': run_names,
                        'Outcome_R2_vs_Prevalence': r2_scores,
                    }).to_csv(per_run_r2_path, index=False)
                    r2_summary_records.append({
                        'Fold': fold_id,
                        'Mean_Outcome_R2_vs_Prevalence': mean_r2,
                        'SD_Outcome_R2_vs_Prevalence': std_r2,
                        'Valid_Runs': len(valid_r2_scores),
                    })

                r2_summary_path = (
                    f'{cal_plot_dir}/R2_Summary_{metric_name}_{model_name}_'
                    f'{self.instability_type}_{test_name}.csv'
                )
                pd.DataFrame(r2_summary_records).to_csv(
                    r2_summary_path,
                    index=False
                )
            print()
                            
    # Fairness metric: subgroup false positive rates and their max-min gap.
    def demographic_false_positive_rate(self):
        metric_name = 'DFPR'

        print('Metric: Race-specific FPR and Maximum FPR Difference (DFPR)')
        for test_name in TEST_NAMES:
            for model_name in MODEL_NAMES:
                dfpr_dir = f'{self.metrics_dir}/dfpr/{model_name}'
                create_nested_directory(dfpr_dir)

                mean_values_by_fold = {}
                std_values_by_fold = {}

                for fold_id in range(1, self.no_of_folds + 1):
                    fold_csv_file_path = (
                        f'{dfpr_dir}/{metric_name}_{model_name}_'
                        f'{self.instability_type}_{test_name}_Fold_{fold_id}.csv'
                    )
                    test_df = self.X_val_list[fold_id - 1] if test_name == 'Validation' else self.X_test
                    y_eval = self.get_y_values(test_name, fold_id)
                    y_true = np.asarray(y_eval).ravel()
                    predictions = self.get_predictions(
                        model_name,
                        test_name,
                        fold_id=fold_id
                    )

                    race_masks = {
                        race: (test_df[race].to_numpy() == 1)
                        for race in RACE_NAMES
                    }
                    negative_counts = {
                        race: int(np.sum(y_true[mask] == 0))
                        for race, mask in race_masks.items()
                    }
                    run_columns = list(predictions.columns)
                    race_rates = {race: [] for race in RACE_NAMES}
                    run_disparities = []

                    for run_index, column_name in enumerate(run_columns, start=1):
                        print(
                            f'{test_name}, {model_name}, Fold '
                            f'{fold_id}/{self.no_of_folds} - Getting DFPRs... '
                            f'{round((run_index / len(run_columns)) * 100)}%',
                            end='\r'
                        )
                        y_pred = predictions[column_name].to_numpy() >= DFPR_THRESHOLD
                        rates_for_run = {}

                        for race in RACE_NAMES:
                            mask = race_masks[race]
                            y_group = y_true[mask]
                            pred_group = y_pred[mask]
                            actual_negative = y_group == 0
                            negative_count = int(np.sum(actual_negative))

                            dfpr = (
                                float(np.mean(pred_group[actual_negative]))
                                if negative_count > 0
                                else np.nan
                            )
                            race_rates[race].append(dfpr)
                            if not np.isnan(dfpr):
                                rates_for_run[race] = dfpr

                        if len(rates_for_run) >= 2:
                            delta_fpr = max(rates_for_run.values()) - min(rates_for_run.values())
                        else:
                            delta_fpr = np.nan
                        run_disparities.append({
                            'Run': column_name,
                            'Max-Min Delta FPR': delta_fpr,
                            'Groups Compared': len(rates_for_run),
                        })

                    print(
                        f'{test_name}, {model_name}, Fold '
                        f'{fold_id}/{self.no_of_folds} - Getting DFPRs... 100%'
                    )

                    rates_df = pd.DataFrame({
                        'Race': RACE_NAMES,
                        'Non-Recidivists': [negative_counts[race] for race in RACE_NAMES],
                        **{
                            column_name: [race_rates[race][run_index] for race in RACE_NAMES]
                            for run_index, column_name in enumerate(run_columns)
                        },
                    })
                    rates_df.to_csv(fold_csv_file_path, index=False)

                    delta_csv_file_path = (
                        f'{dfpr_dir}/Delta_FPR_{model_name}_'
                        f'{self.instability_type}_{test_name}_Fold_{fold_id}.csv'
                    )
                    pd.DataFrame(run_disparities).to_csv(delta_csv_file_path, index=False)

                    fold_means = {}
                    fold_stds = {}
                    for race in RACE_NAMES:
                        valid_scores = np.asarray(race_rates[race], dtype=float)
                        valid_scores = valid_scores[~np.isnan(valid_scores)]
                        fold_means[race] = (
                            float(np.mean(valid_scores)) if len(valid_scores) else np.nan
                        )
                        fold_stds[race] = (
                            float(np.std(valid_scores, ddof=1))
                            if len(valid_scores) > 1 else np.nan
                        )

                    valid_disparities = np.asarray(
                        [row['Max-Min Delta FPR'] for row in run_disparities],
                        dtype=float
                    )
                    valid_disparities = valid_disparities[~np.isnan(valid_disparities)]
                    fold_means['Max-Min Delta FPR'] = (
                        float(np.mean(valid_disparities))
                        if len(valid_disparities) else np.nan
                    )
                    fold_stds['Max-Min Delta FPR'] = (
                        float(np.std(valid_disparities, ddof=1))
                        if len(valid_disparities) > 1 else np.nan
                    )
                    mean_values_by_fold[f'fold_{fold_id}'] = fold_means
                    std_values_by_fold[f'fold_{fold_id}'] = fold_stds

                summary_rows = RACE_NAMES + ['Max-Min Delta FPR']
                mean_df = pd.DataFrame({'Race': summary_rows})
                std_df = pd.DataFrame({'Race': summary_rows})
                for fold_id in range(1, self.no_of_folds + 1):
                    fold_column = f'fold_{fold_id}'
                    mean_df[fold_column] = [
                        mean_values_by_fold[fold_column][race]
                        for race in summary_rows
                    ]
                    std_df[fold_column] = [
                        std_values_by_fold[fold_column][race]
                        for race in summary_rows
                    ]

                mean_csv_file_path = (
                    f'{dfpr_dir}/Means_{metric_name}_{model_name}_'
                    f'{self.instability_type}_{test_name}.csv'
                )
                std_csv_file_path = (
                    f'{dfpr_dir}/STD_{metric_name}_{model_name}_'
                    f'{self.instability_type}_{test_name}.csv'
                )
                mean_df.to_csv(mean_csv_file_path, index=False)
                std_df.to_csv(std_csv_file_path, index=False)
            print()
                
    # SHAP ANALYSIS
    # Uses the runtime shap values per fold
    def shap_analysis(self, trained_models=None):
        metric_name = 'SHAP_Analysis'
        
        print('Metric: SHAP Analysis')
        for test_name in TEST_NAMES:
            for model_name in MODEL_NAMES:
                shap_values_dir = f'{self.metrics_dir}/shap_values/{model_name}'
                shap_dir = f'{self.metrics_dir}/shap_analysis/{model_name}'
                create_nested_directory(shap_dir)

                for fold_id in range(1, self.no_of_folds + 1):
                    print(
                        f'{test_name}, {model_name}, Fold '
                        f'{fold_id}/{self.no_of_folds} - Getting SHAP...',
                        end='\r'
                    )
                    shap_values = pd.read_csv(
                        f'{shap_values_dir}/'
                        f'SHAP_{model_name}_{test_name}_Fold_{fold_id}.csv'
                    )
                    run_columns = shap_values.columns.drop(
                        ['defendant_id', 'feature', 'Mean SHAP'],
                        errors='ignore'
                    )
                    shap_mean = shap_values[run_columns].mean(axis=1)

                    output = pd.DataFrame({
                        'defendant_id': shap_values['defendant_id'],
                        'feature': shap_values['feature'],
                        'SHAP_Mean': shap_mean,
                    })
                    output.to_csv(
                        f'{shap_dir}/{metric_name}_{model_name}_'
                        f'{self.instability_type}_{test_name}_Fold_{fold_id}.csv',
                        index=False
                    )
                print(
                    f'{test_name}, {model_name} - '
                    'Getting SHAP... Done'
                )
        print()
