import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import roc_auc_score, roc_curve, auc, brier_score_loss, confusion_matrix
from sklearn.calibration import calibration_curve
from itertools import combinations
from statsmodels.nonparametric.smoothers_lowess import lowess

from create_dir import create_nested_directory

class Metrics:
    def __init__(self, test_dfs, y_true, instability_type, tests, folds):
        self.race_columns = [
            'race:African-American',
            'race:Asian',
            'race:Caucasian',
            'race:Native-American',
            'race:Hispanic',
            'race:Other'
        ]
        self.models = [ 'LR', 'RF', 'LSVM', 'XGB', 'MLP' ]
        self.test_dfs = test_dfs
        self.y_true = y_true
        self.instability_type = instability_type
        self.tests = tests
        self.folds = folds
        
        self.csv_file_path = ''
        self.metric_result = ''
        self.metric_name = ''
        self.metric_directory = ''
    
    def get_predictions(
        self, 
        model_name,
        test_name,
        fold_id,
        with_ids=False, 
        with_raw=False
    ):
        raw_predictions = pd.read_csv(f'predictions/{self.instability_type}/{model_name}_Predictions_{test_name}_{fold_id}.csv')
        predictions = raw_predictions.drop(columns=['defendant_id'])
        
        data = [predictions]
        if with_ids:
            defendant_ids = raw_predictions['defendant_id']
            data.append(defendant_ids)
        if with_raw:
            data.append(raw_predictions)
        return data
    
    # ===== PART OF METRICS =====     
    # Performance metric
    # Gets the roc_auc (GOOD)
    def roc_auc(self):
        self.metric_name = 'ROC_AUC'
        self.metric_directory = 'roc_auc'
        
        for i, test in enumerate(self.tests):
            print(f'[Generating {self.instability_type}-{test} ROC-AUC and STD]')

            for model in self.models:
                                
                for fold_id in range(1, self.folds + 1):
                    [ predictions ] = self.get_predictions(
                        model_name=model,
                        test_name=test,
                        fold_id=fold_id
                    )
                    
                    if fold_id == 1:
                        self.csv_file_path = self.init_csv(
                            column_names=[*predictions.columns, f'Mean_{self.metric_name}', f'STD_{self.metric_name}'],
                            row_count=self.folds,
                            model_name=model,
                            test_name=test
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
        
        for i, test in enumerate(self.tests):
            print(f'[Generating {self.instability_type}-{test} Brier Scores]')
            
            for model in self.models:                
                for fold_id in range(1, self.folds + 1):                    
                    [ predictions ] = self.get_predictions(
                        model_name=model,
                        test_name=test,
                        fold_id=fold_id
                    )
                    
                    if fold_id == 1:
                        self.csv_file_path = self.init_csv(
                            column_names=[*predictions.columns, f'Mean_{self.metric_name}', f'STD_{self.metric_name}'],
                            row_count=self.folds,
                            model_name=model,
                            test_name=test
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
        
        for i, test in enumerate(self.tests):
            print(f'[Generating {self.instability_type}-{test} 95% Stability Interval]')
            
            for model in self.models:         
                self.csv_file_path = self.init_csv(
                    column_names=[
                        f'Mean_{self.metric_name}_Fold_{fold}' 
                        for fold in range(1, self.folds + 1)
                    ],
                    row_count=1,
                    model_name=model,
                    test_name=test
                )      
                for fold_id in range(1, self.folds + 1):
                    [ predictions, defendant_ids ] = self.get_predictions(model, test, fold_id, True)
                    
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
                        f'{self.instability_type} - {model} - Fold {fold_id} - 95% Stability Intervals of Predicted Recidivism Risk'
                    )
                    plt.xticks(rotation=90)
                    plt.tight_layout()
                    plt.savefig(
                        f'metrics/{self.instability_type}/{self.metric_directory}/'
                        f'{self.metric_name}-{model}-{test}-Fold_{fold_id}.png'
                    )
                    plt.show()
                    plt.close()
            
    # Instability metric
    # Mean Absolute Prediction Error (GOOD)
    def mean_absolute_prediction_error(self): 
        self.metric_directory = 'mape'
        
        for test in self.tests:
            print(f'[Generating {self.instability_type}-{test} MAPE]')
            
            for model in self.models:
                mean_mapes = []
                self.metric_name = 'MAPE'
                
                for fold_id in range(1, self.folds + 1):
                    [ predictions, defendant_ids ] = self.get_predictions(
                        model_name=model, 
                        test_name=test, 
                        fold_id=fold_id,
                        with_ids=True
                    )
                    
                    self.csv_file_path = self.init_csv(
                        column_names=['defendant_id', 'MAPE'],
                        row_count=len(defendant_ids),
                        model_name=model,
                        test_name=test,
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
                    column_names=[f'Fold_{i}' for i in range (1, self.folds + 1)],
                    row_count=1,
                    model_name=model,
                    test_name=test
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
        
        for test in self.tests:
            print(f'[Generating {self.instability_type}-{test} Top-K Jaccard]')
            
            for model in self.models:
                jaccard_scores_mean = []
                jaccard_scores_std = []
                
                for fold_id in range(1, self.folds + 1):
                    jaccard_scores = []
                    
                    [ predictions ] = self.get_predictions(
                        model_name=model, 
                        test_name=test, 
                        fold_id=fold_id
                    )
                    
                    self.csv_file_path = self.init_csv(
                        column_names=['Run_1', 'Run_2', 'Jaccard'],
                        row_count=(len(predictions.columns) * (len(predictions.columns) - 1))//2,
                        model_name=model,
                        test_name=test,
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
                    column_names=[f'Mean_{self.metric_name}', f'STD_{self.metric_name}'],
                    row_count=self.folds,
                    model_name=model,
                    test_name=test,
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
        for test in self.tests:
            print(f'[Generating {self.instability_type}-{test} CII]')
            
            for model in self.models:
                mean_ciis = []
                self.metric_name = 'CII'
                
                for fold_id in range(1, self.folds + 1):
                    [ predictions, defendant_ids ] = self.get_predictions(
                        model_name=model, 
                        test_name=test, 
                        fold_id=fold_id,
                        with_ids=True
                    )
                    
                    self.csv_file_path = self.init_csv(
                        column_names=['defendant_id', 'CII'],
                        row_count=len(defendant_ids),
                        model_name=model,
                        test_name=test,
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
                    plt.title(f'Classification Instability Distribution ({self.instability_type} - {model}, Fold {fold_id})')
                    plt.savefig(
                        f'metrics/{self.instability_type}/{self.metric_directory}/'
                        f'{model}_{test}_{self.metric_name}_Distribution_Plot_{fold_id}.png')
                    plt.close()
                
                self.metric_name = 'Mean_CII'
                self.csv_file_path = self.init_csv(
                    column_names=[f'Fold_{i}' for i in range (1, self.folds + 1)],
                    row_count=1,
                    model_name=model,
                    test_name=test
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
        
        for i, test in enumerate(self.tests):
            print(f'[Generating {self.instability_type}-{test} Calibration Plot]')
            
            for model in self.models:
                
                for fold_id in range(1, self.folds + 1):
                    [ predictions ] = self.get_predictions(
                        model_name=model, 
                        test_name=test, 
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
                        f'{self.instability_type} - {model} - {test} - '
                        f'Calibration Plot - Fold {fold_id}'
                    )
                    plt.legend()
                    plt.tight_layout()

                    plt.savefig(
                        f'metrics/{self.instability_type}/{self.metric_directory}/'
                        f'{model}-{test}-{self.metric_name}-Fold_{fold_id}.png',
                        dpi=300,
                        bbox_inches='tight'
                    )
                    plt.close()
                            
    # Fairness metric 
    # Focuses on getting the false positive rate per demographic (race) (GOOD)                   
    def demographic_false_positive_rate(self):
        self.metric_name = 'DFPR'
        self.metric_directory = 'dfpr'

        for i, test in enumerate(self.tests):
            print(f'[Generating {self.instability_type}-{test} DFPR]')

            for model in self.models:
                if test == 'Validation':
                    for fold_id, test_df in enumerate(self.test_dfs, start=1):
                        
                        [predictions] = self.get_predictions(
                            model_name=model,
                            test_name=test,
                            fold_id=fold_id
                        )

                        self.csv_file_path = self.init_csv(
                            column_names=[
                                'Race',
                                *predictions.columns,
                                f'Mean_{self.metric_name}',
                                f'STD_{self.metric_name}'
                            ],
                            row_count=len(self.race_columns),
                            model_name=model,
                            test_name=test,
                            fold_id=fold_id,
                            first_column_data=self.race_columns
                        )

                        for race_row, race in enumerate(self.race_columns):
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
                    test_df = self.test_dfs[0]
                    for fold_id in range(1, self.folds + 1):
                        [predictions] = self.get_predictions(
                            model_name=model,
                            test_name=test,
                            fold_id=fold_id
                        )

                        self.csv_file_path = self.init_csv(
                            column_names=[
                                'Race',
                                *predictions.columns,
                                f'Mean_{self.metric_name}',
                                f'STD_{self.metric_name}'
                            ],
                            row_count=len(self.race_columns),
                            model_name=model,
                            test_name=test,
                            fold_id=fold_id,
                            first_column_data=self.race_columns
                        )

                        for race_row, race in enumerate(self.race_columns):
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
    
    # Reliability metric
    def calculate_mrip(
        self,
        X,
        y,
        instability_type,
        tests,
        folds,
        feature_std,
        feature_names,
        n_perturbations=50,
        epsilon=0.1,
        delta=0.05
    ):  
        for i, test in enumerate(self.tests):
            for model in self.models:
                for fold_id in range(1, self.folds + 1):
                    [ predictions, defendant_ids ] = self.get_predictions(model, test, fold_id, True)
                            
                    mrip_values = []

                    for i, x in enumerate(X):

                        x_perturbed = generate_perturbations(
                            x,
                            feature_std,
                            feature_names,
                            n_perturbations=n_perturbations,
                            sigma_factor=0.05,
                            delta=delta
                        )

                        probabilities = model.predict_proba(
                            x_perturbed
                        )[:, 1]

                        errors = np.abs(
                            self.y_true[i] - probabilities
                        )

                        mrip_value = np.mean(
                            errors <= epsilon
                        )

                        mrip_values.append(mrip_value)

                    return pd.DataFrame({
                        'defendant_id': defendant_ids,
                        'MRIP': mrip_values
                    })

    def shap_analysis():
        print('hello wurl')
            
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