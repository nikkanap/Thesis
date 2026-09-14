import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix

from itertools import combinations
from statsmodels.nonparametric.smoothers_lowess import lowess

from create_dir import create_nested_directory

RACE_NAMES = [
    'race:African-American',
    'race:Asian',
    'race:Caucasian',
    'race:Native-American',
    'race:Hispanic',
    'race:Other'
]
MODEL_NAMES = [ 'LR', 'RF', 'LSVM', 'XGB', 'MLP' ]
TEST_NAMES = [ 'Validation', 'Test'] 

class PostMetrics:
    def __init__(
        self, 
        X_train, y_train,   # Training data (not split)
        X_val_arr, y_val_arr, val_defendant_ids_arr,
        X_test, y_test, test_defendant_ids,     # Testing data  
        no_of_folds,        
        predictions_dir,  
        instability_type
    ):
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
    
    # Getter functions
    # Gets the defendant ids from the test or validation data
    def get_defendant_ids(self, test_name, fold_id=None):
        if test_name == 'Test': # Test defendant_ids
            return self.test_defendant_ids

        # Validation defendant_ids
        return self.val_defendant_ids_arr[fold_id-1]['defendant_id'].values
    
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
        roc_auc_dir = f'{self.metrics_dir}/roc_auc'
        create_nested_directory(roc_auc_dir)
        
        for i, test_name in enumerate(TEST_NAMES):
            print(f'[Generating {self.instability_type}-{test_name} ROC-AUC and STD]')
                        
            for model_name in MODEL_NAMES:
                print(f'MODEL: {model_name}')
                roc_auc_means = []
                roc_auc_stds = []
                
                for fold_id in range(1, self.no_of_folds + 1):
                    y_df = self.get_y_values(test_name, fold_id)
                    
                    by_fold_csv_file_path = f'{roc_auc_dir}/{metric_name}_{model_name}_{self.instability_type}_{test_name}_Fold_{fold_id}.csv'
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
                        y_pred_proba = predictions[col].values
                        
                        roc_auc = roc_auc_score(y_df, y_pred_proba)
                        roc_aucs.append(roc_auc)
                        
                    self.save_to_csv(
                        by_fold_csv_file_path,
                        'roc_auc_score',
                        roc_aucs
                    )
                    
                    roc_auc_mean = np.mean(roc_aucs)
                    roc_auc_means.append(roc_auc_mean)
                                        
                    roc_auc_std = np.std(roc_aucs, ddof=1)
                    roc_auc_stds.append(roc_auc_std)
                
                csv_file_path = f'{roc_auc_dir}/Aggregated_{metric_name}_{model_name}_{self.instability_type}_{test_name}.csv'
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
        
    # Performance metric
    # Gets the brier score (GOOD)
    def brier_score(self):
        metric_name = 'Brier_Score'
        brier_score_dir = f'{self.metrics_dir}/brier_score'
        create_nested_directory(brier_score_dir)
        
        for i, test_name in enumerate(TEST_NAMES):
            print(f'[Generating {self.instability_type}-{test_name} Brier Scores]')
            
            for model_name in MODEL_NAMES:         
                brier_score_means = []
                brier_score_stds = []
                
                for fold_id in range(1, self.no_of_folds + 1):
                    y_df = self.get_y_values(test_name, fold_id)
                                               
                    by_fold_csv_file_path = f'{brier_score_dir}/{metric_name}_{model_name}_{self.instability_type}_{test_name}_Fold_{fold_id}.csv'                               
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
                        y_pred_proba = predictions[col].values
                        
                        brier_score = brier_score_loss(y_df, y_pred_proba)
                        brier_scores.append(brier_score)
                    
                    self.save_to_csv(
                        by_fold_csv_file_path,
                        'brier_score',
                        brier_scores
                    )
                    
                    # Compute mean and STD of Brier scores across runs
                    brier_score_mean = np.mean(brier_scores)
                    brier_score_means.append(brier_score_mean)
                
                    brier_score_std = np.std(brier_scores, ddof=1)
                    brier_score_stds.append(brier_score_std)
                
                csv_file_path = f'{brier_score_dir}/Aggregated_{metric_name}_{model_name}_{self.instability_type}_{test_name}.csv'           
                self.save_to_csv(
                    csv_file_path,
                    'fold',
                    [i for i in range(1, self.no_of_folds + 1)]                   
                )
                
                self.save_to_csv(
                    csv_file_path,
                    f'Mean_{metric_name}',
                    brier_score_means                   
                )    
                    
                self.save_to_csv(
                    csv_file_path,
                    f'STD_{metric_name}',
                    brier_score_stds
                )

    # Instability metric
    # Gets the 95% Instability Percentile (GOOD)
    def ninety_five_stability_interval(self):
        metric_name = '95_Stability_Interval'
        si_dir = f'{self.metrics_dir}/95_stability_interval'
        create_nested_directory(si_dir)
        
        for i, test_name in enumerate(TEST_NAMES):
            print(f'[Generating {self.instability_type}-{test_name} 95% Stability Interval]')
            mean_si_widths = []
            
            for model_name in MODEL_NAMES:     
                csv_file_path = f'{si_dir}/Aggregated_{metric_name}_{model_name}_{self.instability_type}_{test_name}.csv'
                                    
                for fold_id in range(1, self.no_of_folds + 1):
                                        
                    predictions = self.get_predictions(model_name, test_name, fold_id)
                    defendant_ids = self.get_defendant_ids(test_name, fold_id)
                    
                    stability_df = pd.DataFrame({
                        'defendant_id': defendant_ids,
                        'mean_prediction': np.mean(predictions, axis=1),
                        'lower_95': np.percentile(predictions, 2.5, axis=1),
                        'upper_95': np.percentile(predictions, 97.5, axis=1)
                    })
                    
                    stability_df['si_width'] = ( 
                        stability_df['upper_95'] - stability_df['lower_95']
                    )
                    mean_si_width = stability_df['si_width'].mean()
                    mean_si_widths.append(mean_si_width)
                                        
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
                        f'{self.instability_type} - {model_name} - Fold {fold_id} - 95% Stability Intervals of Predicted Recidivism Risk'
                    )
                    plt.xticks(rotation=90)
                    plt.tight_layout()
                    plt.savefig(csv_file_path.replace('csv','png'))
                    plt.show()
                    plt.close()
                    
                self.save_to_csv(
                    csv_file_path,
                    f'fold',
                    [i for i in range(1, self.no_of_folds + 1)]
                )
                
                self.save_to_csv(
                    csv_file_path,
                    f'Mean_SI_Width',
                    mean_si_widths
                )
            
    # Instability metric
    # Mean Absolute Prediction Error (GOOD)
    def mean_absolute_prediction_error(self): 
        metric_name = 'MAPE'
        mape_dir = f'{self.metrics_dir}/mape'
        create_nested_directory(mape_dir)
        
        for test_name in TEST_NAMES:
            print(f'[Generating {self.instability_type}-{test_name} MAPE]')
            
            for model_name in MODEL_NAMES:
                mean_mapes = []
                
                for fold_id in range(1, self.no_of_folds + 1):
                    predictions = self.get_predictions(
                        model_name, 
                        test_name, 
                        fold_id=fold_id
                    )
                    
                    mape_csv_file_path = f'{mape_dir}/{metric_name}_{model_name}_{self.instability_type}_{test_name}_Fold_{fold_id}.csv'
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
            
    # Instability metric (GOOD)
    def top_k_jaccard(self):
        metric_name = 'Top_K_Jaccard'
        tk_jaccard_dir = f'{self.metrics_dir}/top_k_jaccard'
        create_nested_directory(tk_jaccard_dir)
        
        for test_name in TEST_NAMES:
            print(f'[Generating {self.instability_type}-{test_name} Top-K Jaccard]')
            
            for model_name in MODEL_NAMES:
                jaccard_scores_mean = []
                jaccard_scores_std = []
                
                for fold_id in range(1, self.no_of_folds + 1):
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
                    
                    by_fold_csv_file_path = f'{tk_jaccard_dir}/{metric_name}_{model_name}_{self.instability_type}_{test_name}_Fold_{fold_id}.csv'                        
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
                        
                    mean_jaccard = np.mean(jaccard_scores)
                    jaccard_scores_mean.append(mean_jaccard)
                    
                    std_jaccard = np.std(jaccard_scores, ddof=1)
                    jaccard_scores_std.append(std_jaccard)
                    
                csv_file_path = f'{tk_jaccard_dir}/Aggregate_{metric_name}_{model_name}_{self.instability_type}_{test_name}.csv'
                self.save_to_csv(
                    csv_file_path,
                    'fold',
                    [i for i in range(1, self.no_of_folds + 1)]
                )
                self.save_to_csv(
                    csv_file_path,
                    f'Mean_{metric_name}',
                    jaccard_scores_mean
                )
                self.save_to_csv(
                    csv_file_path,
                    f'STD_{metric_name}',
                    jaccard_scores_std
                )
            
    # Instability metric 
    # CII (GOOD)
    def classification_instability_index(self):
        threshold = 0.5
        metric_name = 'CII'
        cii_dir = f'{self.metrics_dir}/cii'
        create_nested_directory(cii_dir)
        
        for test_name in TEST_NAMES:
            print(f'[Generating {self.instability_type}-{test_name} CII]')
            
            for model_name in MODEL_NAMES:
                mean_ciis = []
                std_ciis = []
                
                for fold_id in range(1, self.no_of_folds + 1):
                    predictions = self.get_predictions(
                        model_name, 
                        test_name, 
                        fold_id=fold_id
                    )
                    
                    defendant_ids = self.get_defendant_ids(test_name, fold_id)
                    
                    by_fold_csv_file_path = f'{cii_dir}/{metric_name}_{model_name}_{self.instability_type}_{test_name}_Fold_{fold_id}.csv'    
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
                                  
                    cii_mean = np.mean(cii_individual)
                    mean_ciis.append(cii_mean)
                    
                    cii_std = np.std(cii_individual, ddof=1)
                    std_ciis.append(cii_std)
                    
                    plt.hist(cii_individual, bins=20)
                    plt.xlabel('CII per Individual')
                    plt.ylabel('Count')
                    plt.title(f'Classification Instability Distribution ({self.instability_type} - {model_name}, Fold {fold_id})')
                    plt.savefig(by_fold_csv_file_path.replace('csv', 'png'))
                    plt.close()
                
                csv_file_path = f'{cii_dir}/Aggregate_{metric_name}_{model_name}_{self.instability_type}_{test_name}.csv' 
                self.save_to_csv(
                    csv_file_path,
                    'fold',
                    [i for i in range (1, self.no_of_folds + 1)]
                )
                
                self.save_to_csv(
                    csv_file_path,
                    f'Mean_{metric_name}',
                    mean_ciis
                )
                
                self.save_to_csv(
                    csv_file_path,
                    f'STD_{metric_name}',
                    std_ciis
                )

    # Performance metric (GOOD)
    def calibration_plot(self):
        metric_name = 'Calibration_Plot'
        cal_plot_dir = f'{self.metrics_dir}/calibration_plot'
        create_nested_directory(cal_plot_dir)
        
        for i, test_name in enumerate(TEST_NAMES):
            print(f'[Generating {self.instability_type}-{test_name} Calibration Plot]')
            
            for model_name in MODEL_NAMES:
                
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
                    plt.legend()
                    plt.tight_layout()

                    plt.savefig(
                        img_file_path,
                        dpi=300,
                        bbox_inches='tight'
                    )
                    plt.close()
                            
    # Fairness metric 
    # Focuses on getting the false positive rate per demographic (race) (GOOD)             
    def demographic_false_positive_rate(self):
        metric_name = 'DFPR'
        dfpr_dir = f'{self.metrics_dir}/dfpr'
        create_nested_directory(dfpr_dir)

        for i, test_name in enumerate(TEST_NAMES):
            print(f'[Generating {self.instability_type}-{test_name} DFPR]')
            
            for model_name in MODEL_NAMES:
                for fold_id in range(1, self.no_of_folds + 1):
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

                    by_fold_csv_file_path = f'{dfpr_dir}/{metric_name}_{model_name}_{self.instability_type}_{test_name}_Fold_{fold_id}.csv'                         
                    self.save_to_csv(
                        by_fold_csv_file_path,
                        'race',
                        RACE_NAMES
                    )
                    
                    # 100 runs
                    for i, column_name in enumerate(predictions.columns):
                        dfpr_scores_by_run = []
                        
                        # per race
                        for race in RACE_NAMES:
                            mask = test_df[race] == 1
                            
                            y_pred = (
                                predictions[column_name].values >= 0.5
                            ).astype(int)

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
                
    
    

   