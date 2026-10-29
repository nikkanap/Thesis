import numpy as np
import pandas as pd
import shutil
import joblib
import os

from models.LR import Logistic_Regression
from models.RF import Random_Forest
from models.XGB import XGBoost
from models.MLP import Multilayer_Perceptron
from models.LSVC import Linear_SVC

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from runtime_metrics import MRIP, get_shap_values
from post_metrics import PostMetrics
from generate_prediction_csv import generate_prediction_csv

RANDOM_SEED = 42
NO_OF_FOLDS = 10
BOOTSTRAPS = 100
INSTABILITY_TYPE = 'Dataset'
MODEL_NAMES = ['LR', 'LSVC', 'XGB', 'RF', 'MLP']

class DatasetInstability:
    def __init__(self, dataset, force_reset=False):      
        print('===== RUNNING DATASET INSTABILITY =====')
        
        # RNG 
        self.rng = np.random.default_rng(RANDOM_SEED)
        self.force_reset = force_reset # forces the program to start from scratch
        
        # K-fold
        self.k_fold = KFold(
            n_splits=NO_OF_FOLDS,
            shuffle=True,
            random_state=RANDOM_SEED
        )

        self.data_dir = f'data/{INSTABILITY_TYPE}'
        if self.force_reset: 
            self.delete_dirs(self.data_dir)
        self.create_nested_directory(self.data_dir)
        
        # Initializing the dataset to training and testing data + ids
        self.X_train = self.get_data(dataset, 'X_train')
        self.y_train = np.asarray(self.get_data(dataset, 'y_train')).ravel()
        self.X_test = self.get_data(dataset, 'X_test')
        self.y_test = np.asarray(self.get_data(dataset, 'y_test')).ravel()
        self.train_ids = np.asarray(self.get_data(dataset, 'train_ids')).ravel()
        self.test_ids = np.asarray(self.get_data(dataset, 'test_ids')).ravel() 
        
        # Save initialized data to csvs (for later recreation)
        self.to_joblib(self.X_train, 'X_train')
        self.to_joblib(self.y_train, 'y_train')
        self.to_joblib(self.X_test, 'X_test')
        self.to_joblib(self.y_test, 'y_test')
        self.to_joblib(self.train_ids, 'train_ids')
        self.to_joblib(self.test_ids, 'test_ids')
        print()

        # Either save the newly generated split data or use existing data
        split_data = None
        if force_reset:
            split_data = self.get_k_split_data()
        self.X_train_split_arr = self.get_data(split_data, 'X_train_split_arr')
        self.y_train_split_arr = self.get_data(split_data, 'y_train_split_arr')
        self.X_val_arr = self.get_data(split_data, 'X_val_arr')
        self.y_val_arr = self.get_data(split_data, 'y_val_arr')
        self.val_ids_fold_arr = self.get_data(split_data, 'val_ids_fold_arr')
        
        # Save initialized array data to csvs (for later recreation)
        self.to_joblib(self.X_train_split_arr, 'X_train_split_arr')
        self.to_joblib(self.y_train_split_arr, 'y_train_split_arr')
        self.to_joblib(self.X_val_arr, 'X_val_arr')
        self.to_joblib(self.y_val_arr, 'y_val_arr')
        self.to_joblib(self.val_ids_fold_arr, 'val_ids_fold_arr')
        print()
        
        bootstrap_data = None
        if force_reset:
            bootstrap_data = self.get_bootstrap_data()
        self.X_boot_arr = self.get_data(bootstrap_data, 'X_boot_arr')
        self.y_boot_arr = self.get_data(bootstrap_data, 'y_boot_arr')
        
        self.to_joblib(self.X_boot_arr, 'X_boot_arr')
        self.to_joblib(self.y_boot_arr, 'y_boot_arr')
                
        [
            self.scaled_X_boot_arr,
            self.transf_X_val_arr,
            self.transf_X_test_arr
        ] = self.get_X_scaled()
        
        self.models_dir = f'models/{INSTABILITY_TYPE}'
        # running the model
        
        self.predictions_dir = f'predictions/{INSTABILITY_TYPE}'
        if force_reset:
            self.train_models()
            self.delete_dirs(self.predictions_dir)
            self.create_nested_directory(self.predictions_dir)
            self.get_predictions()
        self.get_metrics()
    
    def data_exists(self, data_name):
        csv_file_path = f'{self.data_dir}/{data_name}.joblib'
        return os.path.isfile(csv_file_path)
    
    def get_data(self, generated_data, data_name):
        path = f'{self.data_dir}/{data_name}.joblib'

        if self.force_reset or not os.path.isfile(path):
            data = generated_data[data_name]
            joblib.dump(data, path)
            return data

        return joblib.load(path)
    
    def to_joblib(self, data, data_name):
        print(f'Saving data: {data_name}')

        joblib.dump(
            data,
            f'{self.data_dir}/{data_name}.joblib'
        )

    # Creates nested directories
    def create_nested_directory(self, directory_name):
        try:
            os.makedirs(directory_name)
            print(f"Directory '{directory_name}' created successfully.")
        except FileExistsError:
            print(f"Directory '{directory_name}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{directory_name}'.")
        except Exception as e:
            print(f"An error occurred: {e}")
    
    # Deletes directories
    def delete_dirs(self, directory_name):
        try:
            shutil.rmtree(directory_name, ignore_errors=True)
        except Exception as e:
            print(f"An error occurred: {e}")
            
    # ================
    # GETTER FUNCTIONS
    # ================
    
    # Gets the k fold split data
    def get_k_split_data(self):
        print('===== SPLITTING DATA TO K FOLDS =====')

        X_train_split_arr = []
        y_train_split_arr = []
        X_val_arr = []
        y_val_arr = []
        val_ids_fold_arr = []

        for count, (train_idx, val_idx) in enumerate(
            self.k_fold.split(self.X_train), start=1
        ):
            print(f'Splitting.... ({count}/{NO_OF_FOLDS})', end='\r')

            X_train_split_arr.append(self.X_train.iloc[train_idx].copy())
            y_train_split_arr.append(self.y_train[train_idx])

            X_val_arr.append(self.X_train.iloc[val_idx].copy())
            y_val_arr.append(self.y_train[val_idx])

            val_ids_fold_arr.append(self.train_ids[val_idx])

        print(f'Splitting.... ({NO_OF_FOLDS}/{NO_OF_FOLDS})')
        print()
        
        return {
            'X_train_split_arr': X_train_split_arr,
            'y_train_split_arr': y_train_split_arr,
            'X_val_arr': X_val_arr,
            'y_val_arr': y_val_arr,
            'val_ids_fold_arr': val_ids_fold_arr
        }

    # Gets scaled version of the data as an array
    def get_X_scaled(self):
        print()
        print('===== SCALED VERSIONS =====')
        scaled_X_boot_arr = []
        transf_X_val_arr = []
        transf_X_test_arr = []

        for fold_id in range(1, NO_OF_FOLDS + 1):            
            X_boot_fold = self.X_boot_arr[fold_id - 1]
            X_val_fold = self.X_val_arr[fold_id - 1]

            scaled_X_boot_fold = []
            transf_X_val_fold = []
            transf_X_test_fold = []

            for run in range(BOOTSTRAPS):
                scaler = StandardScaler()

                scaled_X_boot = scaler.fit_transform(X_boot_fold[run])
                scaled_X_boot_fold.append(scaled_X_boot)

                transf_X_val = scaler.transform(X_val_fold)
                transf_X_val_fold.append(transf_X_val)

                transf_X_test = scaler.transform(self.X_test)
                transf_X_test_fold.append(transf_X_test)
                print(f'Fold {fold_id}/{NO_OF_FOLDS} - Getting Scaled/Transformed Versions.... {round(((run+1)/BOOTSTRAPS)*100)}%', end="\r")
            print(f'Fold {fold_id}/{NO_OF_FOLDS} - Getting Scaled/Transformed Versions.... 100%')
            
            scaled_X_boot_arr.append(scaled_X_boot_fold)
            transf_X_val_arr.append(transf_X_val_fold)
            transf_X_test_arr.append(transf_X_test_fold)
        print()
        
        return [
            scaled_X_boot_arr,
            transf_X_val_arr,
            transf_X_test_arr
        ]
    
    # Gets the bootstrap training data as an array
    def get_bootstrap_data(self):
        print('===== BOOTSTRAPPING =====')
        X_boot_arr = []
        y_boot_arr = []
        
        for fold_id in range(1, NO_OF_FOLDS+1):            
            X_train_split = self.X_train_split_arr[fold_id-1]
            y_train_split = self.y_train_split_arr[fold_id-1]
            
            X_boot_fold_arr = []
            y_boot_fold_arr = []
            for b in range(1, BOOTSTRAPS + 1):           
                print(
                    f"Fold {fold_id}/{NO_OF_FOLDS} - Getting bootstrapped samples: {round((b/BOOTSTRAPS)*100)}%",
                    end="\r"
                )         
                n_samples = len(X_train_split)
        
                boot_idx = self.rng.choice(
                    n_samples,
                    size=n_samples,
                    replace=True
                )
        
                X_boot = X_train_split.iloc[boot_idx]
                X_boot_fold_arr.append(X_boot)
                
                y_boot = y_train_split[boot_idx]
                y_boot_fold_arr.append(y_boot)
            
            X_boot_arr.append(X_boot_fold_arr)
            y_boot_arr.append(y_boot_fold_arr)
            print(f"Fold {fold_id}/{NO_OF_FOLDS} - Getting bootstrapped samples: 100%")     
        print()  
        
        return {
            'X_boot_arr': X_boot_arr, 
            'y_boot_arr': y_boot_arr 
        }
    
    # Gets the trained models as an array
    def train_models(self):
        print('===== TRAINING MODELS =====')
        
        # Random forest
        Random_Forest(
            self.X_boot_arr,
            self.y_boot_arr,
            NO_OF_FOLDS,
            BOOTSTRAPS,
            INSTABILITY_TYPE, 
            self.models_dir,
            self.force_reset,
            RANDOM_SEED
        )
        
        # XGBoost
        XGBoost(
            self.X_boot_arr,
            self.y_boot_arr,
            NO_OF_FOLDS,
            BOOTSTRAPS,
            INSTABILITY_TYPE, 
            self.models_dir,
            self.force_reset,
            RANDOM_SEED
        )
            
        # Logistic Regression
        Logistic_Regression(
            self.scaled_X_boot_arr,
            self.y_boot_arr,
            NO_OF_FOLDS,
            BOOTSTRAPS,
            INSTABILITY_TYPE, 
            self.models_dir,
            self.force_reset,
            RANDOM_SEED
        )
        
        # Linear SVC (SVM)
        Linear_SVC(
            self.scaled_X_boot_arr,
            self.y_boot_arr,
            NO_OF_FOLDS,
            BOOTSTRAPS,
            INSTABILITY_TYPE, 
            self.models_dir,
            self.force_reset,
            RANDOM_SEED
        )
        
        # Multilayer Perceptron
        Multilayer_Perceptron(
            self.scaled_X_boot_arr,
            self.y_boot_arr,
            NO_OF_FOLDS,
            BOOTSTRAPS,
            INSTABILITY_TYPE, 
            self.models_dir,
            self.force_reset,
            RANDOM_SEED
        )

    # Gets the predictions from the trained models
    def get_predictions(self):
        for model_name in MODEL_NAMES:
            print(f'Model: {model_name}')
            
            for fold_id in range(1, NO_OF_FOLDS+1):
                if model_name in ['LR', 'LSVC', 'MLP']:
                    transf_X_val_fold = self.transf_X_val_arr[fold_id - 1]
                    transf_X_test_fold = self.transf_X_test_arr[fold_id - 1]
                else:
                    X_val_fold = self.X_val_arr[fold_id - 1]
                    X_test_fold = self.X_test

                for b in range(1, BOOTSTRAPS + 1):
                    print(f'Fold {fold_id}/{NO_OF_FOLDS} - Getting model predictions... {round((b/BOOTSTRAPS)*100)}%', end='\r')
                    
                    if model_name in ['LR', 'LSVC', 'MLP']:
                        X_val = transf_X_val_fold[b - 1]
                        X_test = transf_X_test_fold[b - 1]
                    else:
                        X_val = X_val_fold
                        X_test = X_test_fold

                    model = joblib.load(f"{self.models_dir}/{model_name}/Fold_{fold_id}_Bootstrap_{b}.joblib")
                    y_pred_proba = model.predict_proba(X_val)[:, 1]

                    column_name = f'Bootstrap_{b}'
                    csv_file_path = (
                        f'{self.predictions_dir}/'
                        f'{model_name}_Predictions_Validation_{fold_id}.csv'
                    )
                    generate_prediction_csv(
                        y_pred_proba,
                        column_name,
                        csv_file_path
                    )

                    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
                    test_csv_file_path = (
                        f'{self.predictions_dir}/'
                        f'{model_name}_Predictions_Test_{fold_id}.csv'
                    )
                    generate_prediction_csv(
                        y_test_pred_proba,
                        column_name,
                        test_csv_file_path
                    )
                print(f'Fold {fold_id}/{NO_OF_FOLDS} - Getting model predictions... 100%')
            print()
            
    # Gets the metric results  
    def get_metrics(self):
        print('===== METRICS =====')
        pm = PostMetrics(
            self.X_train,
            self.y_train,
            self.X_val_arr,
            self.y_val_arr,
            self.val_ids_fold_arr,
            self.X_test,
            self.y_test,
            self.test_ids,
            NO_OF_FOLDS,
            self.predictions_dir,
            INSTABILITY_TYPE,
            self.force_reset
        )

        pm.roc_auc()
        pm.brier_score()
        pm.calibration_plot()
        pm.ninety_five_prediction_interval()
        pm.mean_absolute_prediction_error()
        pm.classification_instability_index()
        pm.top_k_jaccard()
        pm.demographic_false_positive_rate()
        
        self.get_mrip()
        self.generate_shap_values()
        
        pm.shap_analysis(self.trained_models)
    
    # ===============
    # METRIC-SPECIFIC
    # ===============      
    
    # Gets the mrip       
    def get_mrip(self):
        for model_name in MODEL_NAMES:
            print(f'MRIP for {model_name}')
            
            random_fold = self.get_random_fold()
            random_run = self.get_random_run()
            
            y_train_fold = self.y_boot_arr[random_fold-1]
            val_ids_fold = self.val_ids_fold_arr[random_fold-1]
            X_train_fold = self.X_boot_arr[random_fold-1]
            X_val_fold = self.X_val_arr[random_fold - 1]
            
            X_train = X_train_fold[random_run-1]
            y_train = y_train_fold[random_run-1]
            
            mrip_dir = f'metrics/{INSTABILITY_TYPE}/MRIP/{model_name}'
            self.create_nested_directory(mrip_dir)
        
            file_name = f'MRIP_{model_name}_Validation_Fold_{random_fold}_Run_{random_run}.csv'
            csv_file_path = f'{mrip_dir}/{file_name}'
        
            model = joblib.load(f"{self.models_dir}/{model_name}/Fold_{random_fold}_Bootstrap_{random_run}.joblib")
            print(f'Getting MRIP for Validation...', end='\r')
            MRIP(
                X_train,
                y_train,
                X_val_fold,
                val_ids_fold,
                'Validation',
                model, 
                model_name,
                random_fold,
                random_run,
                csv_file_path
            )
            print(f'Getting MRIP for Validation... Completed')
            
            test_file_name = f'MRIP_{model_name}_Test_Fold_{random_fold}_Run_{random_run}.csv'
            test_csv_file_path = f'{mrip_dir}/{test_file_name}'
            print(f'Getting MRIP for Test...', end='\r')
            MRIP(
                X_train,
                y_train,
                self.X_test,
                self.test_ids,
                'Test',
                model, 
                model_name,
                random_fold,
                random_run,
                test_csv_file_path
            )
            print(f'Getting MRIP for Test... Completed')
        
    def generate_shap_values(self):
        self.trained_models = []
        for model_name in MODEL_NAMES:
            print(f'SHAP values for {model_name}')
            random_fold = self.get_random_fold()
            random_run = self.get_random_run()
            
            self.trained_models.append(
                {
                    'name': model_name,
                    'random_fold': random_fold,
                    'random_run': random_run
                }
            )
            
            val_ids_fold = self.val_ids_fold_arr[random_fold-1]
            if model_name in ['LR', 'LSVC', 'MLP']:
                X_train_fold = self.scaled_X_boot_arr[random_fold-1]
                transf_X_val_fold = self.transf_X_val_arr[random_fold - 1]
                X_val = transf_X_val_fold[random_run-1]
                X_test = self.transf_X_test_arr[random_fold - 1][random_run-1]
            else:
                X_train_fold = self.X_boot_arr[random_fold-1]
                X_val = self.X_val_arr[random_fold - 1]
                X_test = self.X_test
            
            X_train = X_train_fold[random_run-1]    
            feature_names = self.X_train.columns
            model = joblib.load(f"{self.models_dir}/{model_name}/Fold_{random_fold}_Bootstrap_{random_run}.joblib")
            
            shap_dir = f'metrics/{INSTABILITY_TYPE}/shap_values/{model_name}'
            self.create_nested_directory(shap_dir)
        
            file_name = f'SHAP_{model_name}_Validation_Fold_{random_fold}_Run_{random_run}.csv'
            csv_file_path = f'{shap_dir}/{file_name}'
            print(f'Getting SHAP values for Validation...', end='\r')
            get_shap_values(
                X_train,
                X_val,
                feature_names,
                model,
                model_name,
                random_run,
                val_ids_fold,
                csv_file_path
            )
            print(f'Getting SHAP values for Validation... Complete')
            
            test_file_name = f'SHAP_{model_name}_Test_Fold_{random_fold}_Run_{random_run}.csv'
            test_csv_file_path = f'{shap_dir}/{test_file_name}' 
            print(f'Getting SHAP values for Test...', end='\r')
            get_shap_values(
                X_train,
                X_test,
                feature_names,
                model,
                model_name,
                random_run,
                self.test_ids,
                test_csv_file_path
            )
            print(f'Getting SHAP values for Test... Complete')
 
    def get_random_fold(self):
        random_fold = self.rng.choice(NO_OF_FOLDS) + 1
        return random_fold
    
    def get_random_run(self):
        random_run = self.rng.choice(BOOTSTRAPS) + 1
        return random_run