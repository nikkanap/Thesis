import numpy as np
import pandas as pd
import joblib

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

RANDOM_SEED = 42
NO_OF_FOLDS = 10
BOOTSTRAPS = 100
INSTABILITY_TYPE = 'Dataset'
MODEL_NAMES = ['LR', 'LSVC', 'XGB', 'RF', 'MLP']
EVAL_NAMES = ['Validation', 'Test']

class DatasetInstability:
    def __init__(self, dataset, force_reset=False):      
        print('===== RUNNING DATASET INSTABILITY =====')
        
        # RNG 
        self.rng = np.random.default_rng(RANDOM_SEED)
        
        # K-fold
        self.k_fold = KFold(
            n_splits=NO_OF_FOLDS,
            shuffle=True,
            random_state=RANDOM_SEED
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
        self.X_train_split_list = jla.get_data(split_data, 'X_train_split_list')
        self.y_train_split_list = jla.get_data(split_data, 'y_train_split_list')
        self.X_val_list = jla.get_data(split_data, 'X_val_list')
        self.y_val_list = jla.get_data(split_data, 'y_val_list')
        self.val_ids_fold_list = jla.get_data(split_data, 'val_ids_fold_list')
        
        # Save initialized array data to csvs (for later recreation)
        jla.to_joblib(self.X_train_split_list, 'X_train_split_list')
        jla.to_joblib(self.y_train_split_list, 'y_train_split_list')
        jla.to_joblib(self.X_val_list, 'X_val_list')
        jla.to_joblib(self.y_val_list, 'y_val_list')
        jla.to_joblib(self.val_ids_fold_list, 'val_ids_fold_list')
        print()
        
        bootstrap_data = None
        if force_reset:
            bootstrap_data = self.get_bootstrap_data()
            jla.to_joblib(self.X_boot_list, 'X_boot_list')
            jla.to_joblib(self.y_boot_list, 'y_boot_list')
        self.X_boot_list = jla.get_data(bootstrap_data, 'X_boot_list')
        self.y_boot_list = jla.get_data(bootstrap_data, 'y_boot_list')
            
        [
            self.scaled_X_boot_list,
            self.transf_X_val_list,
            self.transf_X_test_list
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

        X_train_split_list = []
        y_train_split_list = []
        X_val_list = []
        y_val_list = []
        val_ids_fold_list = []

        for count, (train_idx, val_idx) in enumerate(
            self.k_fold.split(self.X_train), start=1
        ):
            print(f'Splitting.... ({count}/{NO_OF_FOLDS})', end='\r')

            X_train_split_list.append(self.X_train.iloc[train_idx].copy())
            y_train_split_list.append(self.y_train[train_idx])

            X_val_list.append(self.X_train.iloc[val_idx].copy())
            y_val_list.append(self.y_train[val_idx])

            val_ids_fold_list.append(self.train_ids[val_idx])

        print(f'Splitting.... ({NO_OF_FOLDS}/{NO_OF_FOLDS})')
        print()
        
        return {
            'X_train_split_list': X_train_split_list,
            'y_train_split_list': y_train_split_list,
            'X_val_list': X_val_list,
            'y_val_list': y_val_list,
            'val_ids_fold_list': val_ids_fold_list
        }

    # 🗹 Gets scaled version of the data as an array
    def get_X_scaled(self):
        print()
        print('===== SCALED VERSIONS =====')
        scaled_X_boot_list = []
        transf_X_val_list = []
        transf_X_test_list = []

        for fold_id in range(1, NO_OF_FOLDS + 1):            
            X_boot_fold = self.X_boot_list[fold_id - 1]
            X_val_fold = self.X_val_list[fold_id - 1]

            scaled_X_boot_fold = []
            transf_X_val_fold = []
            transf_X_test_fold = []

            for run in range(BOOTSTRAPS):
                scaler = StandardScaler()

                scaled_X_boot = scaler.fit_transform(X_boot_fold[run])
                transf_X_val = scaler.transform(X_val_fold)
                transf_X_test = scaler.transform(self.X_test)
                
                scaled_X_boot_fold.append(scaled_X_boot)
                transf_X_val_fold.append(transf_X_val)
                transf_X_test_fold.append(transf_X_test)
                
                print(f'Fold {fold_id}/{NO_OF_FOLDS} - Getting Scaled/Transformed Versions.... {round(((run+1)/BOOTSTRAPS)*100)}%', end="\r")
            print(f'Fold {fold_id}/{NO_OF_FOLDS} - Getting Scaled/Transformed Versions.... 100%')
            
            scaled_X_boot_list.append(scaled_X_boot_fold)
            transf_X_val_list.append(transf_X_val_fold)
            transf_X_test_list.append(transf_X_test_fold)
        print()
        
        return [
            scaled_X_boot_list,
            transf_X_val_list,
            transf_X_test_list
        ]
    
    # 🗹 Gets the bootstrap training data as an array 
    def get_bootstrap_data(self):
        print('===== BOOTSTRAPPING =====')
        X_boot_list = []
        y_boot_list = []
        
        for fold_id in range(1, NO_OF_FOLDS+1):            
            X_train_split = self.X_train_split_list[fold_id-1]
            y_train_split = self.y_train_split_list[fold_id-1]
            
            X_boot_fold_list = []
            y_boot_fold_list = []
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
                X_boot_fold_list.append(X_boot)
                
                y_boot = y_train_split[boot_idx]
                y_boot_fold_list.append(y_boot)
            
            X_boot_list.append(X_boot_fold_list)
            y_boot_list.append(y_boot_fold_list)
            
            print(f"Fold {fold_id}/{NO_OF_FOLDS} - Getting bootstrapped samples: 100%")     
        print()  
        
        return {
            'X_boot_list': X_boot_list, 
            'y_boot_list': y_boot_list 
        }
    
    # 🗹 Gets the trained models as an array
    def train_models(self):
        print('===== TRAINING MODELS =====')
        
        # Random forest
        Random_Forest(
            self.X_boot_list,
            self.y_boot_list,
            NO_OF_FOLDS,
            BOOTSTRAPS,
            INSTABILITY_TYPE, 
            self.models_dir,
            force_reset,
            RANDOM_SEED
        )
        
        # XGBoost
        XGBoost(
            self.X_boot_list,
            self.y_boot_list,
            NO_OF_FOLDS,
            BOOTSTRAPS,
            INSTABILITY_TYPE, 
            self.models_dir,
            force_reset,
            RANDOM_SEED
        )
            
        # Logistic Regression
        Logistic_Regression(
            self.scaled_X_boot_list,
            self.y_boot_list,
            NO_OF_FOLDS,
            BOOTSTRAPS,
            INSTABILITY_TYPE, 
            self.models_dir,
            force_reset,
            RANDOM_SEED
        )
        
        # Linear SVC (SVM)
        Linear_SVC(
            self.scaled_X_boot_list,
            self.y_boot_list,
            NO_OF_FOLDS,
            BOOTSTRAPS,
            INSTABILITY_TYPE, 
            self.models_dir,
            force_reset,
            RANDOM_SEED
        )
        
        # Multilayer Perceptron
        Multilayer_Perceptron(
            self.scaled_X_boot_list,
            self.y_boot_list,
            NO_OF_FOLDS,
            BOOTSTRAPS,
            INSTABILITY_TYPE, 
            self.models_dir,
            force_reset,
            RANDOM_SEED
        )

    # 🗹 Gets the predictions from the trained models
    def get_predictions(self):
        for eval_name in EVAL_NAMES:
            for model_name in MODEL_NAMES:
                print(f'Model: {model_name}')
                
                for fold_id in range(1, NO_OF_FOLDS+1):
                    if model_name in ['LR', 'LSVC', 'MLP']:
                        transf_X_eval_fold = (
                            self.transf_X_val_list[fold_id - 1]
                            if eval_name == 'Validation'
                            else self.transf_X_test_list[fold_id - 1]
                        )
                    else:
                        X_eval_fold = (
                            self.X_val_list[fold_id - 1]
                            if eval_name == 'Validation'
                            else self.X_test
                        )
                    
                    for b in range(1, BOOTSTRAPS + 1):
                        print(f'Fold {fold_id}/{NO_OF_FOLDS} - Getting model predictions... {round((b/BOOTSTRAPS)*100)}%', end='\r')
                        
                        X_eval = (
                            transf_X_eval_fold[b - 1]
                            if model_name in ['LR', 'LSVC', 'MLP']
                            else X_eval_fold
                        )

                        model = jla.get_model(self.models_dir, model_name, fold_id, b)
                        y_pred_proba = model.predict_proba(X_eval)[:, 1]
                                            
                        csv_path = (f'{self.predictions_dir}/{model_name}_Predictions_{eval_name}_{fold_id}.csv')
                        csva.save_to_csv(csv_path, y_pred_proba, f'Bootstrap_{b}')
                        
                    print(f'Fold {fold_id}/{NO_OF_FOLDS} - Getting model predictions... 100%')
                print()
            
    # 🗹 Gets the metric results  
    def get_metrics(self, force_reset=False):
        print('===== METRICS =====')
        pm = PostMetrics(
            self.X_train, self.y_train,
            self.X_val_list, self.y_val_list, self.val_ids_fold_list,
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
        chosen_X_test = self.X_test.iloc[[0]]
        
        X_test_reference = self.X_test.iloc[1:]
        y_test_reference = self.y_test[1:]

        for model_name in MODEL_NAMES:
            print(f'MRIP for {model_name}')
            mrip_dir = f'metrics/{INSTABILITY_TYPE}/MRIP/{model_name}'
            dir.create_nested(mrip_dir)

            validation_records = []
            test_records = []

            for fold_id in range(1, NO_OF_FOLDS + 1):
                X_train_runs = self.X_boot_list[fold_id - 1]
                X_val_query = self.X_val_list[fold_id - 1].iloc[[0]]
                val_query_ids = [self.val_ids_fold_list[fold_id - 1][0]]

                for run in range(1, len(X_train_runs) + 1):
                    print(f'MRIP {model_name}: Fold {fold_id}/{NO_OF_FOLDS}, '
                        f'Bootstrap {run}/{len(X_train_runs)}',
                        end='\r'
                    )
                    
                    X_train = X_train_runs[run - 1]
                    model = joblib.load(
                        f'{self.models_dir}/{model_name}/'
                        f'Fold_{fold_id}_Bootstrap_{run}.joblib'
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
                        run
                    ))
                    test_records.extend(MRIP(
                        X_train,
                        X_test_reference,
                        y_test_reference,
                        chosen_X_test,
                        self.test_ids,
                        model,
                        model_name,
                        fold_id,
                        run
                    ))
                print()

            save_mrip_results(
                validation_records,
                f'{mrip_dir}/MRIP_{model_name}_Validation.csv',
                run_label='Bootstrap'
            )
            
            save_mrip_results(
                test_records,
                f'{mrip_dir}/MRIP_{model_name}_Test.csv',
                run_label='Bootstrap'
            )
        
    def generate_shap_values(self):
        self.trained_models = []
        for model_name in MODEL_NAMES:
            print(f'SHAP values for {model_name}')
            
            feature_names = self.X_train.columns
            shap_dir = f'metrics/{INSTABILITY_TYPE}/shap_values/{model_name}'
            dir.create_nested(shap_dir)

            for fold_id in range(1, NO_OF_FOLDS + 1):
                val_id = [self.val_ids_fold_list[fold_id - 1][0]]
                X_val_raw = self.X_val_list[fold_id - 1].iloc[[0]]
                X_test_raw = self.X_test.iloc[[0]]
                
                test_id = [self.test_ids[0]]
                validation_shap_runs = []
                test_shap_runs = []

                for bootstrap_id in range(1, len(self.X_boot_list[fold_id - 1]) + 1):
                    print(
                        f'SHAP {model_name}: Fold {fold_id}/{NO_OF_FOLDS}, '
                        f'Bootstrap {bootstrap_id}',
                        end='\r'
                    )
                    if model_name in ['LR', 'LSVC', 'MLP']:
                        X_train = self.scaled_X_boot_list[fold_id - 1][bootstrap_id - 1]
                        X_val = self.transf_X_val_list[fold_id - 1][bootstrap_id - 1][:1]
                        X_test = self.transf_X_test_list[fold_id - 1][bootstrap_id - 1][:1]
                    else:
                        X_train = self.X_boot_list[fold_id - 1][bootstrap_id - 1]
                        X_val = X_val_raw
                        X_test = X_test_raw

                    model = joblib.load(
                        f'{self.models_dir}/{model_name}/'
                        f'Fold_{fold_id}_Bootstrap_{bootstrap_id}.joblib'
                    )
                    validation_shap_runs.append(get_shap_values(
                        X_train,
                        X_val,
                        feature_names,
                        model,
                        model_name,
                        bootstrap_id,
                        val_id,
                        run_label='Bootstrap'
                    ))
                    test_shap_runs.append(get_shap_values(
                        X_train,
                        X_test,
                        feature_names,
                        model,
                        model_name,
                        bootstrap_id,
                        test_id,
                        run_label='Bootstrap'
                    ))
                    self.trained_models.append({
                        'name': model_name,
                        'fold': fold_id,
                        'bootstrap': bootstrap_id,
                    })

                save_shap_values(
                    validation_shap_runs,
                    f'{shap_dir}/SHAP_{model_name}_Validation_Fold_{fold_id}.csv'
                )
                save_shap_values(
                    test_shap_runs,
                    f'{shap_dir}/SHAP_{model_name}_Test_Fold_{fold_id}.csv'
                )
