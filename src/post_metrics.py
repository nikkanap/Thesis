import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix, r2_score

from itertools import combinations
from statsmodels.nonparametric.smoothers_lowess import lowess

from create_dir import create_nested_directory

RACE_NAMES = [
    'race:African-American',
    'race:Asian',
    'race:Caucasian',
    'race:Native_American',
    'race:Hispanic',
    'race:Other'
]
MODEL_NAMES = [ 'LR', 'RF', 'LSVC', 'XGB', 'MLP' ]
TEST_NAMES = [ 'Validation', 'Test'] 

class PostMetrics:
    def __init__(
        self, 
        X_train, y_train,   # Training data (not split)
        X_val_arr, y_val_arr, val_defendant_ids_arr,
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
        self.X_val_arr = X_val_arr
        self.y_val_arr = y_val_arr
        self.val_defendant_ids_arr = val_defendant_ids_arr
        
        # Directory for predictions
        self.predictions_dir = predictions_dir
        self.instability_type = instability_type
        
        # Consistent Data
        self.no_of_folds = no_of_folds
        
        # Directory for metrics
        
        self.metrics_dir = f'metrics/{self.instability_type}'
        if self.force_reset:
            self.delete_dirs(self.metrics_dir)
    
    def delete_dirs(self, directory_name):
        try:
            shutil.rmtree(directory_name, ignore_errors=True)
        except Exception as e:
            print(f"An error occurred: {e}")
    
    # Getter functions
    # Gets the defendant ids from the test or validation data
    def get_defendant_ids(self, test_name, fold_id=None):
        if test_name == 'Test': # Test defendant_ids
            return self.test_defendant_ids

        # Validation defendant_ids
        return self.val_defendant_ids_arr[fold_id-1]
    
    # Gets the y_values from the test or validation data
    def get_y_values(self, test_name, fold_id=None):
        if test_name == 'Test': # y_test
            return self.y_test
        
        # Validation y values
        return self.y_val_arr[fold_id-1]
                        
    # Gets the predictions per fold
    def get_predictions(self, model_name, test_name, fold_id):
        predictions = pd.read_csv(f'{self.predictions_dir}/{model_name}_Predictions_{test_name}_{fold_id}.csv')
        return predictions
    
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
                               
    
    # ===== PART OF METRICS =====     
    # Performance metric
    # Gets the roc_auc (GOOD)
    def roc_auc(self):
        metric_name = 'ROC_AUC'
        
        print(f'Metric: ROC-AUC')
        for i, test_name in enumerate(TEST_NAMES):
            for model_name in MODEL_NAMES:
                roc_auc_dir = f'{self.metrics_dir}/roc_auc/{model_name}'
                create_nested_directory(roc_auc_dir)
                
                roc_auc_means = []
                roc_auc_stds = []
                
                for fold_id in range(1, self.no_of_folds + 1):
                    by_fold_csv_file_path = f'{roc_auc_dir}/{metric_name}_{model_name}_{self.instability_type}_{test_name}_Fold_{fold_id}.csv'
                    y_df = self.get_y_values(test_name, fold_id)
                    
                    predictions = self.get_predictions(model_name,
                        test_name,
                        fold_id=fold_id
                    )
                        
                    self.save_to_csv(
                        by_fold_csv_file_path,
                        'run',
                        predictions.columns
                    )
                    
                    roc_aucs = []
                    for col in predictions.columns:
                        print(f'{test_name}, {model_name}, Fold {fold_id}/{self.no_of_folds} - Getting ROC-AUCs... {round((len(roc_aucs)/len(predictions.columns))*100)}%', end='\r')
                        
                        y_pred_proba = predictions[col].values
                        
                        roc_auc = roc_auc_score(y_df, y_pred_proba)
                        roc_aucs.append(roc_auc)
                        
                    print(f'{test_name}, {model_name}, Fold {fold_id}/{self.no_of_folds} - Getting ROC-AUCs... 100%')
                        
                    self.save_to_csv(
                        by_fold_csv_file_path,
                        'roc_auc_score',
                        roc_aucs
                    )
                    
                    roc_auc_mean = float(f'{np.mean(roc_aucs):.4g}')
                    roc_auc_means.append(roc_auc_mean)
                                        
                    roc_auc_std = float(f'{np.std(roc_aucs, ddof=1):.4g}')
                    roc_auc_stds.append(roc_auc_std)
                
                csv_file_path = f'{roc_auc_dir}/Aggregated_{metric_name}_{model_name}_{self.instability_type}.csv'
                if test_name == 'Validation':
                    self.save_to_csv(
                        csv_file_path,
                        'fold',
                        [i for i in range(1, self.no_of_folds + 1)]                   
                    )
                
                self.save_to_csv(
                    csv_file_path,
                    f'Mean_{metric_name}',
                    roc_auc_means
                )
                
                self.save_to_csv(
                    csv_file_path,
                    f'STD_{metric_name}',
                    roc_auc_stds
                )
        print()
        
    # Performance metric
    # Gets the brier score (GOOD)
    def brier_score(self):
        metric_name = 'Brier_Score'
        
        print('Metric: Brier Score')
        for i, test_name in enumerate(TEST_NAMES):            
            for model_name in MODEL_NAMES:         
                brier_score_dir = f'{self.metrics_dir}/brier_score/{model_name}'
                create_nested_directory(brier_score_dir)
                
                brier_score_means = []
                brier_score_stds = []
                
                for fold_id in range(1, self.no_of_folds + 1):
                    by_fold_csv_file_path = f'{brier_score_dir}/{metric_name}_{model_name}_{self.instability_type}_{test_name}_Fold_{fold_id}.csv'       
                    y_df = self.get_y_values(test_name, fold_id)
                                                                   
                    predictions = self.get_predictions(
                        model_name,
                        test_name,
                        fold_id=fold_id
                    )
                    
                    self.save_to_csv(
                        by_fold_csv_file_path,
                        'run',
                        predictions.columns
                    )
                    
                    brier_scores = []
                    for col in predictions.columns:
                        print(f'{test_name}, {model_name}, Fold {fold_id}/{self.no_of_folds} - Getting Brier Scores... {round((len(brier_scores)/len(predictions.columns))*100)}%', end='\r')
                        y_pred_proba = predictions[col].values
                        
                        brier_score = brier_score_loss(y_df, y_pred_proba)
                        brier_scores.append(brier_score)
                    print(f'{test_name}, {model_name}, Fold {fold_id}/{self.no_of_folds} - Getting Brier Scores... 100%')
                    self.save_to_csv(
                        by_fold_csv_file_path,
                        'brier_score',
                        brier_scores
                    )
                    
                    # Compute mean and STD of Brier scores across runs
                    brier_score_mean = float(f'{np.mean(brier_scores):.4g}')
                    brier_score_means.append(brier_score_mean)
                
                    brier_score_std = float(f'{np.std(brier_scores, ddof=1):.4g}')
                    brier_score_stds.append(brier_score_std)
                
                csv_file_path = f'{brier_score_dir}/Aggregated_{metric_name}_{model_name}_{self.instability_type}.csv'     
                if test_name == 'Validation':
                    self.save_to_csv(
                        csv_file_path,
                        'fold',
                        [i for i in range(1, self.no_of_folds + 1)]                   
                    )
                
                self.save_to_csv(
                    csv_file_path,
                    f'Mean_{metric_name}_{test_name}',
                    brier_score_means                   
                )    
                    
                self.save_to_csv(
                    csv_file_path,
                    f'STD_{metric_name}_{test_name}',
                    brier_score_stds
                )
        print()

    # Instability metric
    # Gets the 95% Instability Percentile (GOOD)
    def ninety_five_prediction_interval(self):
            metric_name = '95_Prediction_Interval'
            print('Metric: 95 Prediction Interval')
    
            for i, test_name in enumerate(TEST_NAMES):
                for model_name in MODEL_NAMES:
    
                    si_dir = f'{self.metrics_dir}/95_prediction_interval/{model_name}'
                    create_nested_directory(si_dir)
    
                    csv_file_path = (
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
                        lower_95_errors.append(float(f'{lower_95:.4g}'))
                        upper_95_errors.append(float(f'{upper_95:.4g}'))
                        mean_errors.append(float(f'{mean_error:.4g}'))
                        median_errors.append(float(f'{median_error:.4g}'))
    
                    # ------------------------------------------------------
                    # Box plot: Prediction Error per Fold
                    # ------------------------------------------------------
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
    
                    plot_path = csv_file_path.replace(
                        '.csv',
                        '_Prediction_Error_Boxplot.png'
                    )
    
                    plt.savefig(plot_path)
                    plt.close()
    
                    self.save_to_csv(
                        csv_file_path,
                        'Fold',
                        folds
                    )
    
                    self.save_to_csv(
                        csv_file_path,
                        'Lower_95_Error',
                        lower_95_errors
                    )
    
                    self.save_to_csv(
                        csv_file_path,
                        'Upper_95_Error',
                        upper_95_errors
                    )
    
                    self.save_to_csv(
                        csv_file_path,
                        'Mean_Error',
                        mean_errors
                    )
    
                    self.save_to_csv(
                        csv_file_path,
                        'Median_Error',
                        median_errors
                    )
    
                    print(
                        f'{test_name}, {model_name}, '
                        f'Fold {self.no_of_folds}/{self.no_of_folds}'
                    )
    
                    print()

    # Instability metric
    # Mean Absolute Prediction Error (GOOD)
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
                    mape_csv_file_path = f'{mape_dir}/{metric_name}_{model_name}_{self.instability_type}_{test_name}_Fold_{fold_id}.csv'
                            
                    predictions = self.get_predictions(
                        model_name, 
                        test_name, 
                        fold_id=fold_id
                    ).to_numpy()  
                    
                    self.save_to_csv(
                        mape_csv_file_path,
                        'defendant_id',
                        self.get_defendant_ids(test_name, fold_id)
                    )
                        
                    mean_prediction = np.mean(predictions, axis=1) 
                    mape_per_defendant = np.mean(
                        np.abs(predictions - mean_prediction[:, None]),
                        axis=1
                    )
                    
                    self.save_to_csv(
                        mape_csv_file_path,
                        'MAPE',
                        mape_per_defendant
                    )
                    
                    mape_mean = np.mean(mape_per_defendant)
                    mean_mapes.append(mape_mean)
                print(f'{test_name}, {model_name}, Fold {self.no_of_folds}/{self.no_of_folds}')
                
                mean_mape_csv_file_path = f'{mape_dir}/Mean_{metric_name}_{model_name}_{self.instability_type}_{test_name}.csv'
                # Save the fold numbers
                self.save_to_csv(
                    mean_mape_csv_file_path,
                    'folds',
                    [f'fold_{i}' for i in range(1, self.no_of_folds + 1)]
                )
                
                self.save_to_csv(
                    mean_mape_csv_file_path,
                    'Mean_MAPE',
                    mean_mapes,
                )
        print()
            
    # Instability metric (GOOD)
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
                    by_fold_csv_file_path = f'{tk_jaccard_dir}/{metric_name}_{model_name}_{self.instability_type}_{test_name}_Fold_{fold_id}.csv'     
                    run_1_arr = []
                    run_2_arr = []
                    jaccard_scores = []
                    
                    predictions = self.get_predictions(
                        model_name, 
                        test_name, 
                        fold_id=fold_id
                    )
                        
                    k = int(0.10 * len(predictions))
                    for col1, col2 in combinations(predictions.columns, 2):
                        run_1_arr.append(col1)
                        run_2_arr.append(col2)
                        
                        run1 = predictions[col1]
                        run2 = predictions[col2]
                        
                        top_k_1 = run1.nlargest(k).index
                        top_k_2 = run2.nlargest(k).index
                        
                        intersection = len(set(top_k_1) & set(top_k_2))
                        union = len(set(top_k_1) | set(top_k_2))

                        jaccard = intersection / union
                        jaccard_scores.append(jaccard)
                                    
                    self.save_to_csv(
                        by_fold_csv_file_path,
                       'run_1',
                       run_1_arr
                    )
                    self.save_to_csv(
                        by_fold_csv_file_path,
                        'run_2',
                        run_2_arr
                    )
                    self.save_to_csv(
                        by_fold_csv_file_path,
                        'jaccard',
                        jaccard_scores
                    )
                        
                    mean_jaccard = float(f'{np.mean(jaccard_scores):.4g}')
                    jaccard_scores_mean.append(mean_jaccard)
                    
                    std_jaccard = float(f'{np.std(jaccard_scores, ddof=1):.4g}')
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
            
    # Instability metric 
    # CII (GOOD)
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
                    by_fold_csv_file_path = f'{cii_dir}/{metric_name}_{model_name}_{self.instability_type}_{test_name}_Fold_{fold_id}.csv'    
                    predictions = self.get_predictions(
                        model_name, 
                        test_name, 
                        fold_id=fold_id
                    )
                    
                    defendant_ids = self.get_defendant_ids(test_name, fold_id)
                                    
                    self.save_to_csv(
                        by_fold_csv_file_path,
                        'defendant_id',
                        defendant_ids
                    )
                    
                    labels = (predictions >= threshold).astype(int)
                    
                    n_1 = labels.sum(axis=1)
                    n_0 = labels.shape[1] - n_1
                    
                    flips = n_0 * n_1
                    
                    cii_individual = flips / (labels.shape[1] * (labels.shape[1] - 1) / 2)
                    self.save_to_csv(
                        by_fold_csv_file_path,
                        'CII',
                        cii_individual
                    )
                                  
                    cii_mean = float(f'{np.mean(cii_individual):.4g}')
                    mean_ciis.append(cii_mean)
                    
                    cii_std = float(f'{np.std(cii_individual, ddof=1):.4g}')
                    std_ciis.append(cii_std)
                    
                    plt.hist(cii_individual, bins=20)
                    plt.xlabel('CII per Individual')
                    plt.ylabel('Count')
                    plt.title(f'Classification Instability Distribution ({self.instability_type} - {model_name}, Fold {fold_id})')
                    plt.savefig(by_fold_csv_file_path.replace('csv', 'png'))
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

    # Performance metric (GOOD)
    def calibration_plot(self):
        metric_name = 'Calibration_Plot'

        print('Metric: Calbration Plot')
        for i, test_name in enumerate(TEST_NAMES):
            for model_name in MODEL_NAMES:
                cal_plot_dir = f'{self.metrics_dir}/calibration_plot/{model_name}'
                create_nested_directory(cal_plot_dir)
                
                for fold_id in range(1, self.no_of_folds + 1):
                    y_df = self.get_y_values(test_name, fold_id)
                                        
                    img_file_path = f'{cal_plot_dir}/{metric_name}_{model_name}_{self.instability_type}_{test_name}_Fold_{fold_id}.png'
                    predictions = self.get_predictions(
                        model_name, 
                        test_name, 
                        fold_id=fold_id
                    )

                    plt.figure(figsize=(8, 6))
                    for run in range(predictions.shape[1]):
                        print(f'{test_name}, {model_name}, Fold {fold_id}/{self.no_of_folds} - Getting Calibration Plots... {round(((run+1)/predictions.shape[1])*100)}%', end='\r')
                        
                        y_pred = predictions.iloc[:, run]
                        
                        prob_true, prob_pred = calibration_curve(
                            y_df,
                            predictions.iloc[:, run],
                            n_bins=10,
                            strategy='uniform'
                        )

                        plt.plot(
                            prob_pred,
                            prob_true,
                            alpha=0.1
                        )
                        
                        r2 = r2_score(y_df, y_pred)
                        
                    print(f'{test_name}, {model_name}, Fold {fold_id}/{self.no_of_folds} - Getting Calibration Plots... 100%')
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
                    
                    plt.text(
                        0.05, 0.95,
                        f'$R^2$ = {r2:.4f}',
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
        print()
                            
    # Fairness metric 
    # Focuses on getting the false positive rate per demographic (race) (GOOD)             
    def demographic_false_positive_rate(self):
        metric_name = 'DFPR'

        print('Metric: Demographic False Positive Rate (DFPR)')
        for i, test_name in enumerate(TEST_NAMES):
            for model_name in MODEL_NAMES:
                dfpr_dir = f'{self.metrics_dir}/dfpr/{model_name}'
                create_nested_directory(dfpr_dir)
                
                for fold_id in range(1, self.no_of_folds + 1):
                    by_fold_csv_file_path = f'{dfpr_dir}/{metric_name}_{model_name}_{self.instability_type}_{test_name}_Fold_{fold_id}.csv'             
                    test_df = self.X_val_arr[fold_id - 1] if test_name == 'Validation' else self.X_test
                    y_df = self.get_y_values(test_name, fold_id)
                    predictions = self.get_predictions(
                        model_name,
                        test_name,
                        fold_id=fold_id
                    )
                    
                    dfpr_scores_by_race = { race: [] for race in RACE_NAMES }
                    dfpr_means = []
                    dfpr_stds = []            
                    
                    self.save_to_csv(
                        by_fold_csv_file_path,
                        'race',
                        RACE_NAMES
                    )
                    
                    # 100 runs
                    for i, column_name in enumerate(predictions.columns):
                        print(f'{test_name}, {model_name}, Fold {fold_id}/{self.no_of_folds} - Getting DFPRs... {round(((i+1)/len(predictions.columns))*100)}%', end='\r')
                        dfpr_scores_by_run = []
                        
                        # per race
                        for race in RACE_NAMES:
                            mask = test_df[race] == 1
                            
                            y_pred = (predictions[column_name].values >= 0.5).astype(int)

                            if len(y_df[mask]) == 0:
                                dfpr = np.nan
                            else:
                                tn, fp, fn, tp = confusion_matrix(
                                    y_df[mask],
                                    y_pred[mask],
                                    labels=[0, 1]
                                ).ravel()

                                dfpr = (
                                    fp / (fp + tn)
                                    if (fp + tn) > 0
                                    else np.nan
                                )
                            dfpr_scores_by_run.append(dfpr)
                            dfpr_scores_by_race[race].append(dfpr)
                            
                        self.save_to_csv(
                            by_fold_csv_file_path,
                            column_name,
                            dfpr_scores_by_run
                        )
                    print(f'{test_name}, {model_name}, Fold {fold_id}/{self.no_of_folds} - Getting DFPRs... 100%')
                    
                    for race in RACE_NAMES:
                        dfpr_mean = np.nanmean(dfpr_scores_by_race[race])
                        dfpr_means.append(dfpr_mean)
                        
                        dfpr_std = np.nanstd(dfpr_scores_by_race[race], ddof=1)
                        dfpr_stds.append(dfpr_std)
                        
                    mean_csv_file_path = f'{dfpr_dir}/Means_{metric_name}_{model_name}_{self.instability_type}_{test_name}.csv'        
                    std_csv_file_path = f'{dfpr_dir}/STD_{metric_name}_{model_name}_{self.instability_type}_{test_name}.csv'
                    
                    if fold_id == 1: 
                        self.save_to_csv(
                            mean_csv_file_path,
                            'race',
                            RACE_NAMES
                        )
                        
                        self.save_to_csv(
                            std_csv_file_path,
                            'race',
                            RACE_NAMES
                        )
                    
                    self.save_to_csv(
                        mean_csv_file_path,
                        f'fold_{fold_id}',
                        dfpr_means
                    )

                    self.save_to_csv(
                        std_csv_file_path,
                        f'fold_{fold_id}',
                        dfpr_stds
                    )
                    
                self.save_to_csv(
                    mean_csv_file_path,
                    f'z_{fold_id}',
                    dfpr_means
                )

                self.save_to_csv(
                    std_csv_file_path,
                    f'fold_{fold_id}',
                    dfpr_stds
                )
        print()
                
    # SHAP ANALYSIS
    # Uses the runtime shap values per fold
    def shap_analysis(self, trained_models):
        metric_name = 'SHAP_Analysis'
        
        print('Metric: SHAP Analysis')
        for test_name in TEST_NAMES:
            for trained_model in trained_models:
                print(f'{test_name}, {trained_model['name']}, Fold {trained_model['random_fold']}, Run {trained_model['random_run'] if self.instability_type == 'Dataset' else trained_model['random_seed']} - Getting SHAP...', end='\r')
                shap_values_dir = (f'{self.metrics_dir}/shap_values/{trained_model['name']}')
                shap_dir = (f'{self.metrics_dir}/shap_analysis/{trained_model['name']}')
                create_nested_directory(shap_dir)

                if test_name == 'Validation':
                    shap_values = pd.read_csv(
                        f'{shap_values_dir}/'
                        f'SHAP_{trained_model['name']}_Validation_Fold_{trained_model['random_fold']}'
                        f'_{'Run' if self.instability_type == 'Dataset' else 'Random_Seed'}_'
                        f'{trained_model['random_run'] if self.instability_type == 'Dataset' else trained_model['random_seed']}.csv'
                    )
                else:
                    shap_values = pd.read_csv(
                        f'{shap_values_dir}/'
                        f'SHAP_{trained_model['name']}_Test_Fold_{trained_model['random_fold']}'
                        f'_{'Run' if self.instability_type == 'Dataset' else 'Random_Seed'}_'
                        f'{trained_model['random_run'] if self.instability_type == 'Dataset' else trained_model['random_seed']}.csv'
                    )

                defendant_ids = shap_values['defendant_id']
                features = shap_values['feature']
                shap_values = shap_values.drop(
                    columns=['defendant_id', 'feature']
                )

                # Each column = one run
                # Each row = one defendant
                shap_mean = shap_values.mean(axis=1)
                shap_std = shap_values.std(axis=1, ddof=1)

                output = pd.DataFrame({
                    'defendant_id': defendant_ids,
                    'feature': features,
                    'SHAP_Mean': shap_mean,
                    'SHAP_STD': shap_std
                })

                csv_file_path = (
                    f'{shap_dir}/'
                    f'{metric_name}_{trained_model['name']}_'
                    f'{self.instability_type}_{test_name}_'
                    f'Fold_{trained_model['random_fold']}.csv'
                )

                output.to_csv(
                    csv_file_path,
                    index=False
                )
            print(f'{test_name}, {trained_model['name']}, Fold {trained_model['random_fold']}, Run {trained_model['random_run'] if self.instability_type == 'Dataset' else trained_model['random_seed']} - Getting SHAP... Done')
        print()
                
    

   
