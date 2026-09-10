import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import roc_auc_score, roc_curve, auc, brier_score_loss, confusion_matrix
from sklearn.calibration import calibration_curve
from statsmodels.nonparametric.smoothers_lowess import lowess
from top_k_jaccard import top_k_jaccard_all

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
    
    def get_predictions(self, model, test, fold_id, with_ids=False, with_raw=False):
        raw_predictions = pd.read_csv(f'predictions/{self.instability_type}/{model}_Predictions_{test}_{fold_id}.csv')
        predictions = raw_predictions.drop(columns=['defendant_id']).to_numpy()
        
        data = [predictions]
        if with_ids:
            defendant_ids = raw_predictions['defendant_id']
            data.append(defendant_ids)
        if with_raw:
            data.append(raw_predictions)
        return data
    
    # ===== PART OF METRICS =====     
    def roc_auc(self):
        print(f'[Generating {self.instability_type}-{self.test} ROC-AUC and STD]')
        
        for i, test in enumerate(self.tests):
            for model in self.models:
                for fold_id in self.folds:
                    [ predictions, defendant_ids ] = self.get_predictions(model, test, fold_id, True)
                                    
                    roc_aucs = []
                    for col in predictions.columns:
                        y_pred_proba = predictions[col].values
                        
                        roc_auc = roc_auc_score(self.y_true[i], y_pred_proba)
                        self.save_to_csv_by_run(roc_auc, 'roc_auc', 'ROC_AUC', model, test, fold_id, col, defendant_ids)
                        roc_aucs.append(roc_auc)
                
                    roc_auc_mean = np.mean(roc_aucs)
                    self.save_to_csv_by_fold(roc_auc_mean, 'roc_auc_mean', 'Mean_ROC_AUC', model, test, fold_id, defendant_ids)
                    
                    roc_auc_std = np.std(roc_aucs, ddof=1)
                    self.save_to_csv_by_fold(roc_auc_std, 'roc_auc_std', 'STD_ROC_AUC', model, test, fold_id, defendant_ids)
        
    # Performance metric
    # Gets the brier score
    def brier_score(self):
        for i, test in enumerate(self.tests):
            print(f'[Generating {self.instability_type}-{test} Brier Scores]')
            
            for model in self.models:
                for fold_id in self.folds:
                    [ predictions, defendant_ids ] = self.get_predictions(model, test, fold_id, True)
                    brier_scores = []

                    for col in predictions.columns:
                        y_pred_proba = predictions[col].values
                        
                        score = brier_score_loss(self.y_true[i], y_pred_proba)
                        self.save_to_csv_by_run(score, 'brier_score', model, test, fold_id, col, col)
                        brier_scores.append(score)
                    
                    # Compute mean and STD of Brier scores across runs
                    mean_brier = np.mean(brier_scores)
                    self.save_to_csv_by_fold(mean_brier, 'mean_brier_score', model, test, fold_id)
                    
                    std_brier = np.std(brier_scores)
                    self.save_to_csv_by_fold(std_brier, 'std_brier_score',  model, test, fold_id)

    # Instability metric
    # Gets the 95% Instability Percentile
    def ninety_five_instability_percentile(self):
        # index for tests = ['validation', 'test']
        for i, test in enumerate(self.tests):
            print(f'[Generating {self.instability_type}-{test} 95% Instability Percentile]')
            
            for model in self.models:
                for fold_id in self.folds:
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
                    plt.show()
                    plt.savefig(
                        f'metrics/{self.instability_type}/95_percentile/'
                        f'95_Percentile_{model}_{test}_{fold_id}.png'
                    )
            
    # Instability metric
    # Mean Absolute Prediction Error
    def mean_absolute_prediction_error(self): # check notebook lm to get the steps for MAPE values
        for i, test in enumerate(self.tests):
            print(f'[Generating {self.instability_type}-{test} MAPE]')
            for model in self.models:
                for fold_id in self.folds:
                    [ predictions, defendant_ids ] = self.get_predictions(model, test, fold_id, True)
                        
                    mape_per_defendant = np.mean(
                        np.abs(predictions - self.y_true[:, None]),
                        axis=1
                    )
                    self.save_to_csv_by_fold(mape_per_defendant, 'mape', 'MAPE_per_defendant', model, test, fold_id, defendant_ids)
                    
                    mean_mape = np.mean(mape_per_defendant)
                    self.save_to_csv_all_folds(mean_mape, 'mape', 'Mean_MAPE', model, test, fold_id)          
            
    # Instability metric
    def top_k_jaccard(self):
        for i, test in enumerate(self.tests):
            print(f'[Generating {self.instability_type}-{self.test} Top-K Jaccard]')
            for model in self.models:
                for fold_id in self.folds:
                    predictions = self.get_predictions(model, test, fold_id)
                    
                    k = int(0.10 * len(predictions))
                    jaccard_scores = top_k_jaccard_all(predictions, k)
                    self.save_to_csv_by_fold(jaccard_scores, 'jaccard_scores', 'Jaccard_Scores', model, test, fold_id)
                    
                    mean_jaccard = np.mean(jaccard_scores)
                    self.save_to_csv_by_fold(mean_jaccard, 'jaccard_scores', 'Mean_Jaccard', model, test, fold_id)
                    print("Mean Top-K Jaccard:", mean_jaccard)
                                                    
                    std_jaccard = np.std(jaccard_scores, ddof=1)
                    self.save_to_csv_by_fold(std_jaccard, 'jaccard_scores', 'STD_Jaccard', model, test, fold_id)
            
    # Instability metric
    def classification_instability_index(self):
        for i, test in enumerate(self.tests):
            print(f'[Generating {self.instability_type}-{test} CII]')
            for model in self.models:
                for fold_id in self.folds:
                    [ predictions, defendant_ids ] = self.get_predictions(model, test, fold_id, True)
                    
                    threshold = 0.5
                    
                    # Convert to binary labels
                    labels = (predictions >= threshold).astype(int)
                
                    # Compute instability per individual
                    majority_label = np.round(labels.mean(axis=1)).astype(int)
                
                    disagreements = (labels != majority_label[:, None]).sum(axis=1)
                    cii_individual = disagreements / labels.shape[1]
                    self.save_to_csv_by_fold(cii_individual, 'cii', 'CII_per_defendant', model, test, fold_id, defendant_ids)
                                    
                    # Mean CII
                    cii_mean = np.mean(cii_individual)
                    self.save_to_csv_all_folds(cii_mean, 'cii', 'CII_Mean', model, test, fold_id)
                    print(f'{model} CII: {cii_mean:.4f}')
                    
                    plt.hist(cii_individual, bins=20)
                    plt.xlabel('CII per Individual')
                    plt.ylabel('Count')
                    plt.title(f'Classification Instability Distribution ({self.instability_type} - {model}, Fold {fold_id})')
                    plt.savefig(
                        f'metrics/{self.instability_type}/cii/'
                        f'{model}_{test}_CII_Distribution_Plot_{fold_id}.png')
                    plt.close()

    # Performance metric
    def calibration_plot(self):
        for i, test in enumerate(self.tests):
            print(f'[Generating {self.instability_type}-{test} Calibration Plot]')
            for model in self.models:
                for fold_id in self.folds:
                    [ predictions ] = self.get_predictions(model, test, fold_id)

                    plt.figure(figsize=(8, 6))

                    for run in range(predictions.shape[1]):
                        prob_true, prob_pred = calibration_curve(
                            self.y_true[i],
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
                        f'{self.instability_type} - {model} - {test} - '
                        f'Calibration Plot - Fold {fold_id}'
                    )
                    plt.legend()
                    plt.tight_layout()

                    plt.savefig(
                        f'metrics/{self.instability_type}/calibration/'
                        f'{model}_{test}_Calibration_Plot_{fold_id}.png',
                        dpi=300,
                        bbox_inches='tight'
                    )

                    plt.close()
                             
    def demographic_false_positive_rate(self):
        for i in range(2):
            print(f'[Generating {self.instability_type}-{self.tests[i]} DFPR]')
            for model in self.models:
                for race in self.race_columns:
                    mask = self.test_dfs[i] == race
                    
                    for fold_id in self.folds:
                        [ predictions ] = self.get_predictions(model, self.tests[i], fold_id)
                        fpr_by_run = []
                        
                        for run_idx in range(predictions.shape[1]):
                            y_pred = (predictions[:, run_idx] >= 0.5).astype(int)
                            
                            tn, fp, fn, tp = confusion_matrix(
                                self.y_true[i][mask],
                                y_pred[mask],
                                labels=[0, 1]
                            ).ravel()

                            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                            self.save_to_csv_by_run(fpr, 'dfpr', f'DFPR_{race}', model, self.tests[i], fold_id, f'run_{run_idx + 1}')
                            fpr_by_run.append(fpr)
                        
                        mean_fpr = np.mean(fpr_by_run)
                        self.save_to_csv_by_fold(mean_fpr, 'dfpr', f'Mean_DFPR_{race}', model, self.tests[i], fold_id)
                        
                        std_fpr = np.std(fpr_by_run, ddof=1)
                        self.save_to_csv_by_fold(std_fpr, 'dfpr', f'STD_DFPR_{race}', model, self.tests[i], fold_id)
                
    def calculate_mrip(
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
                for fold_id in self.folds:
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

            
    # ===== HELPER FUNCTIONS =====
    # saves results from all k_folds per model in one csv file
    def save_to_csv_all_folds(self, metric_result, metric_directory, metric_name, model_name, test_name, fold_id):
        csv_file_path = f'metrics/{self.instability_type}/{metric_directory}/{metric_name}_{model_name}_{test_name}.csv'
        create_nested_directory(f'metrics/{self.instability_type}/{metric_name}')
        
        if os.path.isfile(csv_file_path):
            results_csv = pd.read_csv(csv_file_path)
            results_csv[f'fold_{fold_id}'] = metric_result
        else:
            results_csv = pd.DataFrame({
                f'fold_{fold_id}': metric_result
            })
        results_csv.to_csv(csv_file_path, index=False)

    # saves results from all runs in a k_fold per model csv file
    def save_to_csv_by_fold(self, metric_result, metric_directory, metric_name, model_name, test_name, fold_id, defendant_ids=None):
        csv_file_path = f'metrics/{self.instability_type}/{metric_directory}/{metric_name}_{model_name}_{test_name}_{fold_id}.csv'
        create_nested_directory(f'metrics/{self.instability_type}/{metric_name}')
        
        if os.path.isfile(csv_file_path):
            results_csv = pd.read_csv(csv_file_path)
            results_csv[metric_name] = metric_result
        else:
            results_csv = pd.DataFrame({
                **({f'defendant_id': defendant_ids} if defendant_ids != None else {}),
                metric_name: metric_result
            })
        results_csv.to_csv(csv_file_path, index=False)

    # saves individual results per run, also in a k_fold per model csv file
    def save_to_csv_by_run(self, metric_result, metric_directory, metric_name, model_name, test_name, fold_id, col, defendant_ids=None):
        csv_file_path = f'metrics/{self.instability_type}/{metric_directory}/{metric_name}_{model_name}_{test_name}_{fold_id}.csv'
        create_nested_directory(f'metrics/{self.instability_type}/{metric_name}')
        
        if os.path.isfile(csv_file_path):
            results_csv = pd.read_csv(csv_file_path)
            results_csv[col] = metric_result
        else:
            results_csv = pd.DataFrame({
                **({f'defendant_id': defendant_ids} if defendant_ids != None else {}),
                col: metric_result
            })
        results_csv.to_csv(csv_file_path, index=False)

    def get_predictions(self, model, test, fold_id, with_ids=False, with_raw=False):
        raw_predictions = pd.read_csv(f'predictions/{self.instability_type}/{model}_Predictions_{test}_{fold_id}.csv')
        predictions = raw_predictions.drop(columns=['defendant_id']).to_numpy()
        
        data = [predictions]
        if with_ids:
            defendant_ids = raw_predictions['defendant_id']
            data.append(defendant_ids)
        if with_raw:
            data.append(raw_predictions)
        return data