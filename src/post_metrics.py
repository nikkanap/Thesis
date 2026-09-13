import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from itertools import combinations
from statsmodels.nonparametric.smoothers_lowess import lowess

from create_dir import create_nested_directory

RACE_COLUMNS = [
    'race:African-American',
    'race:Asian',
    'race:Caucasian',
    'race:Native-American',
    'race:Hispanic',
    'race:Other'
]
MODEL_NAMES = [ 'LR', 'RF', 'LSVM', 'XGB', 'MLP' ]
TEST_NAMES = [ 'Validation', 'Test'] 
INSTABILITY_NAMES = [ '', 'Stochastic_Instability']

class PostMetrics:
    def __init__(
        self, 
        X_train, y_train,   # Training data (not split)
        X_test, y_test,     # Testing data
        validation_indices, train_indices, # indices
        predictions_dir,    
        no_of_folds,        
        bootstrap_indices=None,  # Optional
    ):
        # Official Training data (not split yet to validation and bootstrapped samples)
        self.X_train = X_train
        self.y_train = y_train
        
        # Official Testing  Data
        self.X_test = X_test
        self.y_test = y_test
        
        # Indices (for recreating the training and validation data)
        self.validation_indices = validation_indices
        self.train_indices = train_indices
        self.bootstrap_indices = bootstrap_indices
        
        # Directory for predictions
        self.predictions_dir = predictions_dir
        self.instability_type = f'{'Dataset' if bootstrap_indices != None else 'Stochastic'}_Instability'
        
        # Consistent Data
        self.no_of_folds = no_of_folds
        
        # Variables changed during metric calls
        self.csv_file_path = ''
        self.metric_result = ''
        self.metric_name = ''
        self.metric_directory = ''    
        
    # Recreates the validation data and bootstrapped training data to be used for the metrics
    def recreate_data_by_val_index(
        self, 
        train_idx,
        val_idx
    ):
        # We'll be saving the boot data as one whole array since this function gets data per fold
        X_boot_arr = []
        y_boot_arr = []
        
        # Making the training data separate from the validation data through val_idx and train_idx
        X_train_split = self.X_train.iloc[train_idx]
        y_train_split = self.y_train[train_idx]
        
        val_defendant_ids = self.X_train.iloc[val_idx]['defendant_id'].values # we get this
        X_val = self.X_train.iloc[val_idx]  # we get this
        y_val = self.y_train[val_idx]       # also this
        
        for boot_idx in self.bootstrap_indices:
            X_boot = X_train_split[boot_idx]
            X_boot_arr.append(X_boot)
            
            y_boot = y_train_split[boot_idx]
            y_boot_arr.append(y_boot)
        
        return [X_boot_arr, y_boot_arr, X_val, y_val, val_defendant_ids]
    
    # Getter functions
    def get_validation_defendant_ids(self, val_idx):
        val_defendant_ids = self.X_train.iloc[val_idx]['defendant_id'].values
        return val_defendant_ids

    def get_test_defendant_ids(self):
        test_defendant_ids = self.X_test.iloc['defendant_id'].values
        return test_defendant_ids
    
    def get_predictions(self, model_name, test_name, fold_id):
        predictions = pd.read_csv(f'{self.predictions_dir}/{model_name}_Predictions_{test_name}_{fold_id}.csv')
        return predictions
    
    # ===== PART OF METRICS =====     
    # Performance metric
    # Gets the roc_auc (GOOD)
    def roc_auc(self):
        self.metric_name = 'ROC_AUC'
        self.metric_directory = 'roc_auc'
        
        for i, test_name in enumerate(TEST_NAMES):
            print(f'[Generating {self.instability_type}-{test_name} ROC-AUC and STD]')

            for model_name in MODEL_NAMES:
                print(f'MODEL: {model_name}')
                                
                for fold_id in range(1, self.no_of_folds + 1):
                    predictions = self.get_predictions(model_name,
                        test_name,
                        fold_id=fold_id
                    )
                    
                    if fold_id == 1:
                        self.csv_file_path = self.init_csv(
                            [*predictions.columns, f'Mean_{self.metric_name}', f'STD_{self.metric_name}'],
                            self.no_of_folds,
                            model_name,
                            test_name
                        )
                        
                    roc_aucs = []
                    for col in predictions.columns:
                        y_pred_proba = predictions[col].values
                        
                        roc_auc = roc_auc_score(self.y_true[i], y_pred_proba)
                        self.save_to_csv(
                            metric_result=roc_auc, 
                            metric_column_name=col,
                            row=fold_id - 1
                        )
                        roc_aucs.append(roc_auc)
                    
                    roc_auc_mean = np.mean(roc_aucs)
                    self.save_to_csv(
                        metric_result=roc_auc_mean,
                        metric_column_name=f'Mean_{self.metric_name}',
                        row=fold_id - 1
                    )
                    
                    roc_auc_std = np.std(roc_aucs, ddof=1)
                    self.save_to_csv(
                        metric_result=roc_auc_std,
                        metric_column_name=f'STD_{self.metric_name}',
                        row=fold_id - 1
                    )
        
    # Performance metric
    # Gets the brier score (GOOD)
    def brier_score(self):
        self.metric_name = 'Brier_Score'
        self.metric_directory = 'brier_score'
        
        for i, test_name in enumerate(TEST_NAMES):
            print(f'[Generating {self.instability_type}-{test_name} Brier Scores]')
            
            for model_name in MODEL_NAMES:                
                for fold_id in range(1, self.no_of_folds + 1):                    
                    predictions = self.get_predictions(
                        model_name,
                        test_name,
                        fold_id=fold_id
                    )
                    
                    if fold_id == 1:
                        self.csv_file_path = self.init_csv(
                            [*predictions.columns, f'Mean_{self.metric_name}', f'STD_{self.metric_name}'],
                            self.no_of_folds,
                            model_name,
                            test_name
                        )
                    
                    brier_scores = []
                    for col in predictions.columns:
                        y_pred_proba = predictions[col].values
                        
                        brier_score = brier_score_loss(self.y_true[i], y_pred_proba)
                        self.save_to_csv(
                            metric_result=brier_score,
                            metric_column_name=col,
                            row=fold_id-1
                        )
                        brier_scores.append(brier_score)
                    
                    # Compute mean and STD of Brier scores across runs
                    brier_score_mean = np.mean(brier_scores)
                    self.save_to_csv(
                        metric_result=brier_score_mean,
                        metric_column_name=f'Mean_{self.metric_name}',
                        row=fold_id-1
                    )
                    
                    brier_score_std = np.std(brier_scores, ddof=1)
                    self.save_to_csv(
                        metric_result=brier_score_std,
                        metric_column_name=f'STD_{self.metric_name}',
                        row=fold_id-1
                    )

    # Instability metric
    # Gets the 95% Instability Percentile (GOOD)
    def ninety_five_stability_interval(self):
        self.metric_name = '95_Stability_Interval'
        self.metric_directory = '95_stability_interval'
        
        for i, test_name in enumerate(TEST_NAMES):
            print(f'[Generating {self.instability_type}-{test_name} 95% Stability Interval]')
            
            for model_name in MODEL_NAMES:         
                self.csv_file_path = self.init_csv(
                    [
                        f'Mean_{self.metric_name}_Fold_{fold}' 
                        for fold in range(1, self.no_of_folds + 1)
                    ],
                    1,
                    model_name,
                    test_name
                )      
                for fold_id in range(1, self.no_of_folds + 1):
                    predictions = self.get_predictions(model_name, test_name, fold_id)
                    
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
                    self.save_to_csv(
                        metric_result=mean_si_width,
                        metric_column_name=f'Mean_{self.metric_name}_Fold_{fold_id}'
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
                        f'{self.instability_type} - {model_name} - Fold {fold_id} - 95% Stability Intervals of Predicted Recidivism Risk'
                    )
                    plt.xticks(rotation=90)
                    plt.tight_layout()
                    plt.savefig(
                        f'metrics/{self.instability_type}/{self.metric_directory}/'
                        f'{self.metric_name}-{model_name}-{test_name}-Fold_{fold_id}.png'
                    )
                    plt.show()
                    plt.close()
            
    # Instability metric
    # Mean Absolute Prediction Error (GOOD)
    def mean_absolute_prediction_error(self): 
        self.metric_directory = 'mape'
        
        for test_name in TEST_NAMES:
            print(f'[Generating {self.instability_type}-{test_name} MAPE]')
            
            for model_name in MODEL_NAMES:
                mean_mapes = []
                self.metric_name = 'MAPE'
                
                for fold_id in range(1, self.no_of_folds + 1):
                    predictions = self.get_predictions(
                        model_name, 
                        test_name, 
                        fold_id=fold_id,
                        with_ids=True
                    )
                    
                    self.csv_file_path = self.init_csv(
                        ['defendant_id', 'MAPE'],
                        len(defendant_ids),
                        model_name,
                        test_name,
                        fold_id=fold_id,
                        first_column_data=defendant_ids
                    )
                        
                    mean_prediction = np.mean(predictions, axis=1)    
                    mape_per_defendant = np.mean(
                        np.abs(predictions - mean_prediction[:, None]),
                        axis=1
                    )
                    
                    for row, mape in enumerate(mape_per_defendant):
                        self.save_to_csv(
                            metric_result=mape,
                            metric_column_name='MAPE',
                            row=row
                        )
                    
                    mape_mean = np.mean(mape_per_defendant)
                    mean_mapes.append(mape_mean)
                
                self.metric_name = 'Mean_MAPE'
                self.csv_file_path = self.init_csv(
                    [f'Fold_{i}' for i in range (1, self.no_of_folds + 1)],
                    1,
                    model_name,
                    test
                )  
                
                for col, column_mean_mape in enumerate(mean_mapes, start=1):
                    self.save_to_csv(
                        metric_result=column_mean_mape,
                        metric_column_name=f'Fold_{col}'
                    )
            
    # Instability metric (GOOD)
    def top_k_jaccard(self):
        self.metric_name = 'Top_K_Jaccard'
        self.metric_directory = 'top_k_jaccard'
        
        for test_name in TEST_NAMES:
            print(f'[Generating {self.instability_type}-{test_name} Top-K Jaccard]')
            
            for model_name in MODEL_NAMES:
                jaccard_scores_mean = []
                jaccard_scores_std = []
                
                for fold_id in range(1, self.no_of_folds + 1):
                    jaccard_scores = []
                    
                    predictions = self.get_predictions(
                        model_name, 
                        test_name, 
                        fold_id=fold_id
                    )
                    
                    self.csv_file_path = self.init_csv(
                        ['Run_1', 'Run_2', 'Jaccard'],
                        (len(predictions.columns) * (len(predictions.columns) - 1))//2,
                        model_name,
                        test_name,
                        fold_id=fold_id
                    )
                        
                    k = int(0.10 * len(predictions))
                    for row, (col1, col2) in enumerate(combinations(predictions.columns, 2)):
                        run1 = predictions[col1]
                        run2 = predictions[col2]
                        
                        top_k_1 = run1.nlargest(k).index
                        top_k_2 = run2.nlargest(k).index
                        
                        intersection = len(set(top_k_1) & set(top_k_2))
                        union = len(set(top_k_1) | set(top_k_2))

                        jaccard = intersection / union
                        self.save_to_csv(
                            metric_result=jaccard,
                            metric_column_name='Jaccard',
                            row=row
                        )
                        self.save_to_csv(
                            metric_result=col1,
                            metric_column_name='Run_1',
                            row=row
                        )
                        self.save_to_csv(
                            metric_result=col2,
                            metric_column_name='Run_2',
                            row=row
                        )
                        jaccard_scores.append(jaccard)
                    
                    mean_jaccard = np.mean(jaccard_scores)
                    jaccard_scores_mean.append(mean_jaccard)
                    
                    std_jaccard = np.std(jaccard_scores, ddof=1)
                    jaccard_scores_std.append(std_jaccard)
                    
                self.csv_file_path = self.init_csv(
                    [f'Mean_{self.metric_name}', f'STD_{self.metric_name}'],
                    self.no_of_folds,
                    model_name,
                    test_name,
                )
                
                for row, (mean, std) in enumerate(zip(jaccard_scores_mean, jaccard_scores_std)):
                    self.save_to_csv(
                        metric_result=mean,
                        metric_column_name=f'Mean_{self.metric_name}',
                        row=row
                    )
                    self.save_to_csv(
                        metric_result=std,
                        metric_column_name=f'STD_{self.metric_name}',
                        row=row
                    )
            
    # Instability metric (GOOD)
    def classification_instability_index(self):
        threshold = 0.5
        
        self.metric_directory = 'cii'
        for test_name in TEST_NAMES:
            print(f'[Generating {self.instability_type}-{test_name} CII]')
            
            for model_name in MODEL_NAMES:
                mean_ciis = []
                self.metric_name = 'CII'
                
                for fold_id in range(1, self.no_of_folds + 1):
                    predictions = self.get_predictions(
                        model_name, 
                        test_name, 
                        fold_id=fold_id,
                        with_ids=True
                    )
                    
                    self.csv_file_path = self.init_csv(
                        ['defendant_id', 'CII'],
                        len(defendant_ids),
                        model_name,
                        test_name,
                        fold_id=fold_id,
                        first_column_data=defendant_ids
                    )
                    
                    labels = (predictions >= threshold).astype(int)
                    
                    n_1 = labels.sum(axis=1)
                    n_0 = labels.shape[1] - n_1
                    
                    flips = n_0 * n_1
                    
                    cii_individual = flips / (labels.shape[1] * (labels.shape[1] - 1) / 2)
                    
                    for row, cii in enumerate(cii_individual):
                        self.save_to_csv(
                            metric_result=cii,
                            metric_column_name='CII',
                            row=row
                        )
                                  
                    cii_mean = np.mean(cii_individual)
                    mean_ciis.append(cii_mean)
                    
                    plt.hist(cii_individual, bins=20)
                    plt.xlabel('CII per Individual')
                    plt.ylabel('Count')
                    plt.title(f'Classification Instability Distribution ({self.instability_type} - {model_name}, Fold {fold_id})')
                    plt.savefig(
                        f'metrics/{self.instability_type}/{self.metric_directory}/'
                        f'{model_name}_{test_name}_{self.metric_name}_Distribution_Plot_{fold_id}.png')
                    plt.close()
                
                self.metric_name = 'Mean_CII'
                self.csv_file_path = self.init_csv(
                    [f'Fold_{i}' for i in range (1, self.no_of_folds + 1)],
                    1,
                    model_name,
                    test_name
                )  
                
                for col, column_mean_cii in enumerate(mean_ciis, start=1):
                    self.save_to_csv(
                        metric_result=column_mean_cii,
                        metric_column_name=f'Fold_{col}'
                    )

    # Performance metric (GOOD)
    def calibration_plot(self):
        self.metric_name = 'Calibration_Plot'
        self.metric_directory = 'calibration_plot'
        
        for i, test_name in enumerate(TEST_NAMES):
            print(f'[Generating {self.instability_type}-{test_name} Calibration Plot]')
            
            for model_name in MODEL_NAMES:
                
                for fold_id in range(1, self.no_of_folds + 1):
                    predictions = self.get_predictions(
                        model_name, 
                        test_name, 
                        fold_id=fold_id
                    )

                    plt.figure(figsize=(8, 6))

                    for run in range(predictions.shape[1]):
                        prob_true, prob_pred = calibration_curve(
                            self.y_true[i],
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
                        f'metrics/{self.instability_type}/{self.metric_directory}/'
                        f'{model_name}-{test_name}-{self.metric_name}-Fold_{fold_id}.png',
                        dpi=300,
                        bbox_inches='tight'
                    )
                    plt.close()
                            
    # Fairness metric 
    # Focuses on getting the false positive rate per demographic (race) (GOOD)                   
    def demographic_false_positive_rate(self):
        self.metric_name = 'DFPR'
        self.metric_directory = 'dfpr'

        test_dfs = self.X[1,3]
        for i, test_name in enumerate(TEST_NAMES):
            print(f'[Generating {self.instability_type}-{test_name} DFPR]')

            for model_name in MODEL_NAMES:
                if test_name == 'Validation':
                    for fold_id, test_df in enumerate(test_dfs, start=1):
                        
                        [predictions] = self.get_predictions(
                            model_name,
                            test_name,
                            fold_id=fold_id
                        )

                        self.csv_file_path = self.init_csv(
                            [
                                'Race',
                                *predictions.columns,
                                f'Mean_{self.metric_name}',
                                f'STD_{self.metric_name}'
                            ],
                            len(RACE_COLUMNS),
                            model_name,
                            test_name,
                            fold_id=fold_id,
                            first_column_data=RACE_COLUMNS
                        )

                        for race_row, race in enumerate(RACE_COLUMNS):
                            mask = test_df[race] == 1

                            dfpr_scores = []

                            for run in predictions.columns:

                                y_pred = (
                                    predictions[run].values >= 0.5
                                ).astype(int)

                                tn, fp, fn, tp = confusion_matrix(
                                    self.y_true[i][mask],
                                    y_pred[mask],
                                    labels=[0, 1]
                                ).ravel()

                                dfpr = (
                                    fp / (fp + tn)
                                    if (fp + tn) > 0
                                    else 0
                                )

                                self.save_to_csv(
                                    metric_result=dfpr,
                                    metric_column_name=run,
                                    row=race_row
                                )

                                dfpr_scores.append(dfpr)

                            mean_dfpr = np.mean(dfpr_scores)
                            std_dfpr = np.std(dfpr_scores, ddof=1)

                            self.save_to_csv(
                                metric_result=mean_dfpr,
                                metric_column_name=f'Mean_{self.metric_name}',
                                row=race_row
                            )

                            self.save_to_csv(
                                metric_result=std_dfpr,
                                metric_column_name=f'STD_{self.metric_name}',
                                row=race_row
                            )
                else:
                    test_df = test_dfs[0]
                    for fold_id in range(1, self.no_of_folds + 1):
                        [predictions] = self.get_predictions(
                            model_name,
                            test_name,
                            fold_id=fold_id
                        )

                        self.csv_file_path = self.init_csv(
                            [
                                'Race',
                                *predictions.columns,
                                f'Mean_{self.metric_name}',
                                f'STD_{self.metric_name}'
                            ],
                            len(RACE_COLUMNS),
                            model_name,
                            test_name,
                            fold_id=fold_id,
                            first_column_data=RACE_COLUMNS
                        )

                        for race_row, race in enumerate(RACE_COLUMNS):
                            mask = test_df[race] == 1

                            dfpr_scores = []

                            for run in predictions.columns:

                                y_pred = (
                                    predictions[run].values >= 0.5
                                ).astype(int)

                                tn, fp, fn, tp = confusion_matrix(
                                    self.y_true[i][mask],
                                    y_pred[mask],
                                    labels=[0, 1]
                                ).ravel()

                                dfpr = (
                                    fp / (fp + tn)
                                    if (fp + tn) > 0
                                    else 0
                                )

                                self.save_to_csv(
                                    metric_result=dfpr,
                                    metric_column_name=run,
                                    row=race_row
                                )

                                dfpr_scores.append(dfpr)

                            mean_dfpr = np.mean(dfpr_scores)
                            std_dfpr = np.std(dfpr_scores, ddof=1)

                            self.save_to_csv(
                                metric_result=mean_dfpr,
                                metric_column_name=f'Mean_{self.metric_name}',
                                row=race_row
                            )

                            self.save_to_csv(
                                metric_result=std_dfpr,
                                metric_column_name=f'STD_{self.metric_name}',
                                row=race_row
                            )       
                
    # ===== HELPER FUNCTIONS =====
    def init_csv(
        self,
        column_names,
        row_count,
        model_name,
        test_name,
        fold_id=None,
        first_column_data=None,
    ):
        file_directory = f'metrics/{self.instability_type}/{self.metric_name}'
        create_nested_directory(file_directory)
        
        if first_column_data is None:
            df = pd.DataFrame(
                index=[f"{i}" for i in range(1, row_count + 1)],
                columns=column_names
            )
        else:
            df = pd.DataFrame(
                {column_names[0]: first_column_data},
                columns=column_names
            )
        fold_name = f'-Fold_{fold_id}' if fold_id != None else ''
        file_name = f'{self.metric_name}-{model_name}-{test_name}{fold_name}.csv'
        csv_file_path = f'{file_directory}/{file_name}'
        df.to_csv(csv_file_path, index=False)
        
        return csv_file_path
    
    def save_to_csv(
        self, 
        metric_result,
        metric_column_name,
        row=None
    ):
        if os.path.isfile(self.csv_file_path):
            results_csv = pd.read_csv(self.csv_file_path)
            
            if row is None:
                results_csv[metric_column_name] = metric_result
            else:
                results_csv.loc[row, metric_column_name] = metric_result
        results_csv.to_csv(self.csv_file_path, index=False)
  
    
    
    

   