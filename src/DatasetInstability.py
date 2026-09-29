import numpy as np
import pandas as pd
import shutil
import os

from models.LR import Logistic_Regression
from models.RF import Random_Forest
from models.XGB import XGBoost
from models.MLP import Multilayer_Perceptron
from models.LSVC import Linear_SVC
from init_database import init_dataset

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from runtime_metrics import MRIP, get_shap_values
from post_metrics import PostMetrics
from generate_prediction_csv import generate_prediction_csv

RANDOM_SEED = 42
NO_OF_FOLDS = 10
BOOTSTRAPS = 100
INSTABILITY_TYPE = 'Dataset'

class DatasetInstability:
    def __init__(self):      
        # RNG 
        self.rng = np.random.default_rng(RANDOM_SEED)
          
        # K-fold
        self.k_fold = KFold(
            n_splits=NO_OF_FOLDS,
            shuffle=True,
            random_state=RANDOM_SEED
        )

        # Initializing the dataset to training and testing data + ids
        [
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.train_ids,
            self.test_ids
        ] = init_dataset()

        # Save initialized data to csvs (for later recreation)
        self.data_dir = f'data/{INSTABILITY_TYPE}'
        self.delete_dirs(self.data_dir)
        self.create_nested_directory(self.data_dir)
        
        self.to_csv(self.X_train, 'X_train')
        self.to_csv(self.y_train, 'y_train', 'y_train')
        self.to_csv(self.X_test, 'X_test')
        self.to_csv(self.y_test, 'y_test', 'y_test')
        self.to_csv(self.train_ids, 'train_ids', 'defendant_id')
        self.to_csv(self.test_ids, 'test_ids', 'defendant_id')
        
        [   
            self.X_train_split_arr,
            self.y_train_split_arr,
            self.X_val_arr,
            self.y_val_arr,
            self.val_ids_fold_arr,
        ] = self.get_k_split_data()
        
        # Save initialized array data to csvs (for later recreation)
        self.to_csv(self.X_train_split_arr, 'X_train_split_arr')
        self.to_csv(self.y_train_split_arr, 'y_train_split_arr', 'y_train')
        self.to_csv(self.X_val_arr, 'X_val_arr')
        self.to_csv(self.y_val_arr, 'y_val_arr')
        self.to_csv(self.val_ids_fold_arr, 'val_ids_fold_arr')
        
        [
            self.X_boot_arr,
            self.y_boot_arr
        ] = self.get_bootstrap_data()

        [
            self.scaled_X_boot_arr,
            self.transf_X_val_arr,
            self.transf_X_test_arr
        ] = self.get_X_scaled()
        
        # running the model
        self.trained_models = self.get_trained_models()
        
        self.predictions_dir = f'predictions/{INSTABILITY_TYPE}'
        self.delete_dirs(self.predictions_dir)
        self.create_nested_directory(self.predictions_dir)
        self.get_predictions()        
        self.get_metrics()
    
    # Saves data to csvs
    def to_csv(self, data, data_name, column_name=None):
        if isinstance(data, pd.DataFrame):
            print(f'Saving data: {data_name}')
            data.to_csv(
                f'{self.data_dir}/{data_name}.csv',
                index=False
            )

        elif isinstance(data, np.ndarray):
            pd.DataFrame({
                column_name: data
            }).to_csv(
                f'{self.data_dir}/{data_name}.csv',
                index=False
            )

        elif isinstance(data, list):
            if all(isinstance(x, pd.DataFrame) for x in data):
                pd.concat(
                    data,
                    keys=range(1, len(data) + 1),
                    names=['fold_id']
                ).to_csv(
                    f'{self.data_dir}/{data_name}.csv'
                )

            elif all(isinstance(x, np.ndarray) for x in data):
                pd.concat(
                    [pd.Series(x) for x in data],
                    axis=1
                ).T.to_csv(
                    f'{self.data_dir}/{data_name}.csv',
                    index=False
                )

            else:
                print('Invalid data. Cannot save to csv.')

        else:
            print('Invalid data. Cannot save to csv.')

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
        
        count = 1
        for train_idx, val_idx in self.k_fold.split(self.X_train):
            print(f'\nSplitting.... ({count}/{NO_OF_FOLDS})', end="\r")
            
            X_train_split = self.X_train.iloc[train_idx]
            X_train_split_arr.append(X_train_split)
            
            y_train_split = self.y_train[train_idx]
            y_train_split_arr.append(y_train_split)

            X_val = self.X_train.iloc[val_idx]
            X_val_arr.append(X_val)
            
            y_val = self.y_train[val_idx]
            y_val_arr.append(y_val)

            val_ids_fold = self.train_ids[val_idx]
            val_ids_fold_arr.append(val_ids_fold)
            
            count += 1
        
        return [ X_train_split_arr, y_train_split_arr, X_val_arr, y_val_arr, val_ids_fold_arr ]

    # Gets scaled version of the data as an array
    def get_X_scaled(self):
        print('===== SCALED VERSIONS =====')
        scaled_X_boot_arr = []
        transf_X_val_arr = []
        transf_X_test_arr = []

        for fold_id in range(1, NO_OF_FOLDS + 1):
            print(f'\nGetting Scaled/Transformed Versions.... ({fold_id}/{NO_OF_FOLDS})', end="\r")
            
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

            scaled_X_boot_arr.append(scaled_X_boot_fold)
            transf_X_val_arr.append(transf_X_val_fold)
            transf_X_test_arr.append(transf_X_test_fold)
            
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
            print(f'Fold {fold_id}/{NO_OF_FOLDS}')
            
            X_train_split = self.X_train_split_arr[fold_id-1]
            y_train_split = self.y_train_split_arr[fold_id-1]
            
            X_boot_fold_arr = []
            y_boot_fold_arr = []
            for b in range(BOOTSTRAPS):    
                print(f'\nGetting Bootstrap Data.... ({b}/{BOOTSTRAPS})', end="\r")
                
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
            
        return [ X_boot_arr, y_boot_arr ]
    
    # Gets the trained models as an array
    def get_trained_models(self):
        print('===== TRAINING MODELS =====')
        
        # Random forest
        rf_trained_models = Random_Forest(
            self.X_boot_arr,
            self.y_boot_arr,
            NO_OF_FOLDS,
            BOOTSTRAPS,
            INSTABILITY_TYPE, 
            random_seed=RANDOM_SEED
        )
        
        # XGBoost
        xgb_trained_models = XGBoost(
            self.X_boot_arr,
            self.y_boot_arr,
            NO_OF_FOLDS,
            BOOTSTRAPS,
            INSTABILITY_TYPE, 
            random_seed=RANDOM_SEED
        )
            
        # Logistic Regression
        lr_trained_models = Logistic_Regression(
            self.scaled_X_boot_arr,
            self.y_boot_arr,
            NO_OF_FOLDS,
            BOOTSTRAPS,
            INSTABILITY_TYPE, 
            random_seed=RANDOM_SEED
        )
        
        # Linear SVC (SVM)
        lsvc_trained_models = Linear_SVC(
            self.scaled_X_boot_arr,
            self.y_boot_arr,
            NO_OF_FOLDS,
            BOOTSTRAPS,
            INSTABILITY_TYPE, 
            random_seed=RANDOM_SEED
        )
        
        # Multilayer Perceptron
        mlp_trained_models = Multilayer_Perceptron(
            self.scaled_X_boot_arr,
            self.y_boot_arr,
            NO_OF_FOLDS,
            BOOTSTRAPS,
            INSTABILITY_TYPE, 
            random_seed=RANDOM_SEED
        )
            
        return [ 
            {   
                'name': 'RF',
                'models': rf_trained_models
            },
            {
                'name': 'XGB',
                'models': xgb_trained_models
            },
            {
                'name': 'LR',
                'models': lr_trained_models
            },
            {
                'name': 'LSVC',
                'models': lsvc_trained_models
            },
            {
                'name': 'MLP',
                'models': mlp_trained_models   
            }
        ]

    # Gets the predictions from the trained models
    def get_predictions(self):
        for trained_model in self.trained_models:
            print(f'Model: {trained_model['name']}')
            
            for fold_id, models_by_fold in enumerate(trained_model['models'],start=1):
                print(f'Fold_{fold_id}')

                if trained_model['name'] in ['LR', 'LSVC', 'MLP']:
                    transf_X_val_fold = self.transf_X_val_arr[fold_id - 1]
                    transf_X_test_fold = self.transf_X_test_arr[fold_id - 1]
                else:
                    X_val_fold = self.X_val_arr[fold_id - 1]
                    X_test_fold = self.X_test

                for nth_run, model in enumerate(models_by_fold, start=1):
                    print(f'Run_{nth_run}')
                    
                    if trained_model['name'] in ['LR', 'LSVC', 'MLP']:
                        X_val = transf_X_val_fold[nth_run - 1]
                        X_test = transf_X_test_fold[nth_run - 1]
                    else:
                        X_val = X_val_fold
                        X_test = X_test_fold

                    y_pred_proba = model.predict_proba(X_val)[:, 1]

                    column_name = f'Bootstrap_{nth_run}'
                    csv_file_path = (
                        f'{self.predictions_dir}/'
                        f'{trained_model["name"]}_Predictions_Validation_{fold_id}.csv'
                    )
                    generate_prediction_csv(
                        y_pred_proba,
                        column_name,
                        csv_file_path
                    )

                    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
                    test_csv_file_path = (
                        f'{self.predictions_dir}/'
                        f'{trained_model["name"]}_Predictions_Test_{fold_id}.csv'
                    )
                    generate_prediction_csv(
                        y_test_pred_proba,
                        column_name,
                        test_csv_file_path
                    )
                  
    # Gets the metric results  
    def get_metrics(self):
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
            INSTABILITY_TYPE
        )

        pm.roc_auc()
        pm.brier_score()
        pm.calibration_plot()
        pm.ninety_five_stability_interval()
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
        for trained_model in self.trained_models:
            print(f'MRIP for {trained_model["name"]}')
            
            random_fold = self.get_random_fold()
            random_run = self.get_random_run()
            print(f'Fold_{random_fold}')
            print(f'Run_{random_run}')
            
            y_train_fold = self.y_boot_arr[random_fold-1]
            val_ids_fold = self.val_ids_fold_arr[random_fold-1]
            X_train_fold = self.X_boot_arr[random_fold-1]
            X_val_fold = self.X_val_arr[random_fold - 1]
            
            X_train = X_train_fold[random_run-1]
            y_train = y_train_fold[random_run-1]
            
            mrip_dir = f'metrics/{INSTABILITY_TYPE}/MRIP/{trained_model["name"]}'
            self.create_nested_directory(mrip_dir)
        
            file_name = f'MRIP_{trained_model["name"]}_Validation_Fold_{random_fold}_Run_{random_run}.csv'
            csv_file_path = f'{mrip_dir}/{file_name}'
            
            
            model = trained_model['models'][random_fold-1][random_run-1]
            model_name = trained_model["name"]
            
            MRIP(
                X_train,
                y_train,
                X_val_fold,
                val_ids_fold,
                model, 
                model_name,
                random_run,
                csv_file_path
            )
            
            test_file_name = f'MRIP_{trained_model["name"]}_Test_Fold_{random_fold}_Run_{random_run}.csv'
            test_csv_file_path = f'{mrip_dir}/{test_file_name}'
            
            MRIP(
                X_train,
                y_train,
                self.X_test,
                self.test_ids,
                model, 
                model_name,
                random_run,
                test_csv_file_path
            )
        
    def generate_shap_values(self):
        for trained_model in self.trained_models:
            print(f'SHAP for {trained_model["name"]}')
            
            random_fold = self.get_random_fold()
            trained_model['random_fold'] = random_fold
            print(f'Fold_{random_fold}')
            
            random_run = self.get_random_run()
            trained_model['random_run'] = random_run
            print(f'Run_{random_run}')
            
            val_ids_fold = self.val_ids_fold_arr[random_fold-1]
            if trained_model['name'] in ['LR', 'LSVC', 'MLP']:
                print('if LR LSVC or MLP')
                X_train_fold = self.scaled_X_boot_arr[random_fold-1]
                transf_X_val_fold = self.transf_X_val_arr[random_fold - 1]
                X_val = transf_X_val_fold[random_run-1]
                X_test = self.transf_X_test_arr[random_fold - 1][random_run-1]
            else:
                print('else XGB RF')
                X_train_fold = self.X_boot_arr[random_fold-1]
                X_val = self.X_val_arr[random_fold - 1]
                X_test = self.X_test
            
            X_train = X_train_fold[random_run-1]    
            feature_names = self.X_train.columns
            model = trained_model['models'][random_fold-1][random_run-1]
            model_name = trained_model['name']
            
            shap_dir = f'metrics/{INSTABILITY_TYPE}/shap_values/{trained_model["name"]}'
            self.create_nested_directory(shap_dir)
        
            file_name = f'SHAP_{trained_model["name"]}_Validation_Fold_{random_fold}_Run_{random_run}.csv'
            csv_file_path = f'{shap_dir}/{file_name}'
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
            
            test_file_name = f'SHAP_{trained_model["name"]}_Test_Fold_{random_fold}_Run_{random_run}.csv'
            test_csv_file_path = f'{shap_dir}/{test_file_name}' 
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
 
    def get_random_fold(self):
        random_fold = self.rng.choice(NO_OF_FOLDS) + 1
        return random_fold
    
    def get_random_run(self):
        random_run = self.rng.choice(BOOTSTRAPS) + 1
        return random_run