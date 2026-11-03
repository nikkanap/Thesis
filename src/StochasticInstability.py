import numpy as np
import pandas as pd
import joblib
import os

from models.LR import Logistic_Regression
from models.RF import Random_Forest
from models.XGB import XGBoost
from models.MLP import Multilayer_Perceptron
from models.LSVC import Linear_SVC

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from runtime_metrics import (
    MRIP,
    get_shap_values,
    save_mrip_results,
    save_shap_values,
)
from post_metrics import PostMetrics
import csv_actions as csva
import joblib_actions as jla

import directory as dir

RANDOM_SEEDS = 10
NO_OF_FOLDS = 10
INSTABILITY_TYPE = 'Stochastic'
MODEL_NAMES = ['LR', 'LSVC', 'XGB', 'RF', 'MLP']
EVAL_NAMES = ['Validation', 'Test']

class StochasticInstability:
    def __init__(self, dataset=None, force_reset=False):   
        print('===== RUNNING STOCHASTIC INSTABILITY =====')   
        
        # RNG 
        self.rng = np.random.default_rng(42)
        
        # K-fold
        self.k_fold = KFold(
            n_splits=NO_OF_FOLDS,
            shuffle=True,
            random_state=42
        )
        
        self.data_dir = f'data/{INSTABILITY_TYPE}'
        if force_reset: 
            dir.delete_nested(self.data_dir)
            
        dir.create_nested(self.data_dir)
        
        # Initializing the dataset to training and testing data + ids
        self.X_train = jla.get_data(dataset, 'X_train')
        self.y_train = np.asarray(jla.get_data(dataset, 'y_train')).ravel()
        self.X_test = jla.get_data(dataset, 'X_test')
        self.y_test = np.asarray(jla.get_data(dataset, 'y_test')).ravel()
        self.train_ids = np.asarray(jla.get_data(dataset, 'train_ids')).ravel()
        self.test_ids = np.asarray(jla.get_data(dataset, 'test_ids')).ravel()    
        
        # Save initialized data to csvs (for later recreation)
        jla.to_joblib(self.X_train, 'X_train')
        jla.to_joblib(self.y_train, 'y_train')
        jla.to_joblib(self.X_test, 'X_test')
        jla.to_joblib(self.y_test, 'y_test')
        jla.to_joblib(self.train_ids, 'train_ids')
        jla.to_joblib(self.test_ids, 'test_ids')
        print()
    
        # Either save the newly generated split data or use existing data
        split_data = None
        if force_reset:
            split_data = self.get_k_split_data()
        self.X_train_split_arr = jla.get_data(split_data, 'X_train_split_arr')
        self.y_train_split_arr = jla.get_data(split_data, 'y_train_split_arr')
        self.X_val_arr = jla.get_data(split_data, 'X_val_arr')
        self.y_val_arr = jla.get_data(split_data, 'y_val_arr')
        self.val_ids_fold_arr = jla.get_data(split_data, 'val_ids_fold_arr')
        
        # Save initialized array data to csvs (for later recreation)
        jla.to_joblib(self.X_train_split_arr, 'X_train_split_arr')
        jla.to_joblib(self.y_train_split_arr, 'y_train_split_arr')
        jla.to_joblib(self.X_val_arr, 'X_val_arr')
        jla.to_joblib(self.y_val_arr, 'y_val_arr')
        jla.to_joblib(self.val_ids_fold_arr, 'val_ids_fold_arr')
        print()

        [
            self.scaled_X_train_split_arr,
            self.transf_X_val_arr,
            self.transf_X_test_arr
        ] = self.get_X_scaled()
        
        self.models_dir = f'models/{INSTABILITY_TYPE}'
        
        # running the model
        self.predictions_dir = f'predictions/{INSTABILITY_TYPE}'
        if force_reset:
            self.train_models()
            dir.delete_nested(self.predictions_dir)
            dir.create_nested(self.predictions_dir)
            self.get_predictions()
        self.get_metrics(force_reset)
            
    # ================
    # GETTER FUNCTIONS
    # ================
    
    # 🗹 Gets the k fold split data
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
            print(f'Splitting.... ({count}/{NO_OF_FOLDS})', end="\r")
            
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

    # 🗹 Gets scaled version of the data as an array
    def get_X_scaled(self):
        print('===== SCALED VERSIONS =====')
        scaled_X_train_split_arr = []
        transf_X_val_arr = []
        transf_X_test_arr = []

        for fold_id in range(1, NO_OF_FOLDS + 1):
            X_train_fold = self.X_train_split_arr[fold_id - 1]
            X_val_fold = self.X_val_arr[fold_id - 1]

            scaler = StandardScaler()

            scaled_X_train = scaler.fit_transform(X_train_fold)
            transf_X_val = scaler.transform(X_val_fold)
            transf_X_test = scaler.transform(self.X_test)
            
            scaled_X_train_split_arr.append(scaled_X_train)
            transf_X_val_arr.append(transf_X_val)
            transf_X_test_arr.append(transf_X_test)

            print(f'Fold {fold_id}/{NO_OF_FOLDS}', end="\r")
        print(f'Fold {NO_OF_FOLDS}/{NO_OF_FOLDS}')
        print()
               
        return [
            scaled_X_train_split_arr,
            transf_X_val_arr,
            transf_X_test_arr
        ]
    
    # 🗹 Gets the trained models as an array
    def train_models(self):
        print('===== TRAINING MODELS =====')
        
        # Random forest
        Random_Forest(
            self.X_train_split_arr,
            self.y_train_split_arr,
            NO_OF_FOLDS,
            RANDOM_SEEDS,
            INSTABILITY_TYPE, 
            self.models_dir,
            force_reset
        )
        
        # XGBoost
        XGBoost(
            self.X_train_split_arr,
            self.y_train_split_arr,
            NO_OF_FOLDS,
            RANDOM_SEEDS,
            INSTABILITY_TYPE,
            self.models_dir,
            force_reset
        )
            
        # Logistic Regression
        Logistic_Regression(
            self.scaled_X_train_split_arr,
            self.y_train_split_arr,
            NO_OF_FOLDS,
            RANDOM_SEEDS,
            INSTABILITY_TYPE, 
            self.models_dir,
            force_reset
        )
        
        # Linear SVC (SVM)
        Linear_SVC(
            self.scaled_X_train_split_arr,
            self.y_train_split_arr,
            NO_OF_FOLDS,
            RANDOM_SEEDS,
            INSTABILITY_TYPE, 
            self.models_dir,
            force_reset
        )
        
        # Multilayer Perceptron
        Multilayer_Perceptron(
            self.scaled_X_train_split_arr,
            self.y_train_split_arr,
            NO_OF_FOLDS,
            RANDOM_SEEDS,
            INSTABILITY_TYPE, 
            self.models_dir,
            force_reset
        )

    # 🗹 sGets the predictions from the trained models
    def get_predictions(self):
        for eval_name in EVAL_NAMES:
            for model_name in MODEL_NAMES:
                print(f'Model: {model_name}')
                
                for fold_id in range(1, NO_OF_FOLDS + 1):
                    if model_name in ['LR', 'LSVC', 'MLP']:
                        X_eval = (
                            self.transf_X_val_arr[fold_id - 1]
                            if eval_name == 'Validation'
                            else self.transf_X_test_arr[fold_id - 1]
                        )
                    else:
                        X_eval = (
                            self.X_val_arr[fold_id - 1]
                            if eval_name == 'Validation'
                            else self.X_test
                        )

                    for random_seed in range(1, RANDOM_SEEDS + 1):
                        print(f'Fold {fold_id}/{NO_OF_FOLDS} - Getting predictions... {round((random_seed/RANDOM_SEEDS)*100)}%', end='\r')
                        
                        model = jla.get_model(self.models_dir, model_name, fold_id, random_seed)
                        y_pred_proba = model.predict_proba(X_eval)[:, 1]

                        column_name = f'Random_Seed_{random_seed}'
                        csv_path = (f'{self.predictions_dir}/{model_name}_Predictions_{eval_name}_{fold_id}.csv')
                        
                        csva.save_to_csv(csv_path, y_pred_proba, column_name)
                    print(f'Fold {fold_id}/{NO_OF_FOLDS} - Getting predictions... 100%')
                print()
                  
    # 🗹 Gets the metric results  
    def get_metrics(self, force_reset=False):
        pm = PostMetrics(
            self.X_train, self.y_train,
            self.X_val_arr, self.y_val_arr, self.val_ids_fold_arr,
            self.X_test, self.y_test, self.test_ids,
            NO_OF_FOLDS,
            self.predictions_dir,
            INSTABILITY_TYPE,
            force_reset
        )
        
        pm.roc_auc()
        pm.brier_score()
        pm.calibration_plot()
        pm.ninety_five_prediction_error()
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
        X_test_reference = self.X_test.iloc[1:]
        y_test_reference = self.y_test[1:]
        X_test_query = self.X_test.iloc[[0]]
        test_query_ids = [self.test_ids[0]]

        for model_name in MODEL_NAMES:
            print(f'MRIP for {model_name}')
            mrip_dir = f'metrics/{INSTABILITY_TYPE}/MRIP/{model_name}'
            dir.create_nested(mrip_dir)

            validation_records = []
            test_records = []

            for fold_id in range(1, NO_OF_FOLDS + 1):
                X_train = self.X_train_split_arr[fold_id - 1]
                X_val_query = self.X_val_arr[fold_id - 1].iloc[[0]]
                val_query_ids = [self.val_ids_fold_arr[fold_id - 1][0]]

                for seed_id in range(1, RANDOM_SEEDS + 1):
                    print(
                        f'MRIP {model_name}: Fold {fold_id}/{NO_OF_FOLDS}, '
                        f'Seed {seed_id}/{RANDOM_SEEDS}',
                        end='\r'
                    )
                    model = joblib.load(
                        f'{self.models_dir}/{model_name}/'
                        f'Fold_{fold_id}_Random_Seed_{seed_id}.joblib'
                    )

                    validation_records.extend(MRIP(
                        X_train,
                        self.X_test,
                        self.y_test,
                        X_val_query,
                        val_query_ids,
                        model,
                        model_name,
                        fold_id,
                        seed_id
                    ))
                    test_records.extend(MRIP(
                        X_train,
                        X_test_reference,
                        y_test_reference,
                        X_test_query,
                        test_query_ids,
                        model,
                        model_name,
                        fold_id,
                        seed_id
                    ))

            save_mrip_results(
                validation_records,
                f'{mrip_dir}/MRIP_{model_name}_Validation.csv',
                run_label='Random Seed'
            )
            save_mrip_results(
                test_records,
                f'{mrip_dir}/MRIP_{model_name}_Test.csv',
                run_label='Random Seed'
            )
        
    def generate_shap_values(self):
        self.trained_models = []
        for model_name in MODEL_NAMES:
            print(f'SHAP for {model_name}')
            feature_names = self.X_train.columns
            shap_dir = f'metrics/{INSTABILITY_TYPE}/shap_values/{model_name}'
            dir.create_nested(shap_dir)

            for fold_id in range(1, NO_OF_FOLDS + 1):
                val_id = [self.val_ids_fold_arr[fold_id - 1][0]]
                X_val_raw = self.X_val_arr[fold_id - 1].iloc[[0]]
                X_test_raw = self.X_test.iloc[[0]]
                test_id = [self.test_ids[0]]
                validation_shap_runs = []
                test_shap_runs = []

                for seed_id in range(1, RANDOM_SEEDS + 1):
                    print(
                        f'SHAP {model_name}: Fold {fold_id}/{NO_OF_FOLDS}, '
                        f'Random seed {seed_id}/{RANDOM_SEEDS}',
                        end='\r'
                    )
                    if model_name in ['LR', 'LSVC', 'MLP']:
                        X_train = self.scaled_X_train_split_arr[fold_id - 1]
                        X_val = self.transf_X_val_arr[fold_id - 1][:1]
                        X_test = self.transf_X_test_arr[fold_id - 1][:1]
                    else:
                        X_train = self.X_train_split_arr[fold_id - 1]
                        X_val = X_val_raw
                        X_test = X_test_raw

                    model = joblib.load(
                        f'{self.models_dir}/{model_name}/'
                        f'Fold_{fold_id}_Random_Seed_{seed_id}.joblib'
                    )
                    validation_shap_runs.append(get_shap_values(
                        X_train,
                        X_val,
                        feature_names,
                        model,
                        model_name,
                        seed_id,
                        val_id,
                        run_label='Random_Seed'
                    ))
                    test_shap_runs.append(get_shap_values(
                        X_train,
                        X_test,
                        feature_names,
                        model,
                        model_name,
                        seed_id,
                        test_id,
                        run_label='Random_Seed'
                    ))
                    self.trained_models.append({
                        'name': model_name,
                        'fold': fold_id,
                        'random_seed': seed_id,
                    })

                save_shap_values(validation_shap_runs, f'{shap_dir}/SHAP_{model_name}_Validation_Fold_{fold_id}.csv')
                save_shap_values(test_shap_runs, f'{shap_dir}/SHAP_{model_name}_Test_Fold_{fold_id}.csv')

