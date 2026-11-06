import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from models.LR import Logistic_Regression
from models.RF import Random_Forest
from models.XGB import XGBoost
from models.MLP import Multilayer_Perceptron
from models.LSVC import Linear_SVC

# Metrics
from post_metrics import PostMetrics
import runtime_metrics as rm

# Helper imports
import csv_actions as csva
import joblib_actions as jla
import directory as dir

# Constant variables
FIXED_RANDOM_SEED = 42
NO_OF_FOLDS = 3
NO_OF_BOOTSTRAPS = 100
INSTABILITY_TYPE = 'Dataset'

# Constant lists
MODEL_NAMES = ['LR', 'LSVC', 'XGB', 'RF', 'MLP']
EVAL_NAMES = ['Validation', 'Test']

class DatasetInstability:
    def __init__(self, dataset, force_reset=False):      
        print('\n===== RUNNING DATASET INSTABILITY =====')
        
        # RNG 
        self.rng = np.random.default_rng(FIXED_RANDOM_SEED)
        
        # K-fold
        self.k_fold = KFold(
            n_splits=NO_OF_FOLDS,
            shuffle=True,
            random_state=FIXED_RANDOM_SEED
        )

        self.data_dir = f'data/{INSTABILITY_TYPE}'
        if force_reset: 
            dir.clear_nested(self.data_dir)
            
            self.init_data_dict(dataset, force_reset)
            
            split_data = self.get_k_split_data()
            self.init_data_dict(split_data, force_reset)
            
            bootstrap_data = self.get_bootstrap_data()
            self.init_data_dict(bootstrap_data, force_reset)
            
        else:
            # Fetch only the existing data
            keys = ['X_train', 'y_train', 
                    'X_test', 'y_test', 
                    'train_ids', 'test_ids',
                    'X_train_split_list', 'y_train_split_list',
                    'X_val_list', 'y_val_list',
                    'val_ids_fold_list',
                    'X_boot_list','y_boot_list'
                ]
            self.init_data_dict(keys)
        
        [
            self.scaled_X_boot_list,
            self.transf_X_val_list,
            self.transf_X_test_list
        ] = self.get_X_scaled()
        
        self.models_dir = f'models/{INSTABILITY_TYPE}'
        self.predictions_dir = f'predictions/{INSTABILITY_TYPE}'
        
        if force_reset:
            # Train the models
            self.train_models()
            dir.clear_nested(self.predictions_dir)
            
            # Generate model predictions
            self.generate_predictions()
        self.generate_metrics(force_reset)
   
            
    # ================
    # GETTER FUNCTIONS
    # ================
    
    # 🗹 Gets the k fold split data
    def get_k_split_data(self):
        print('\n===== SPLITTING DATA TO K FOLDS =====')

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
        print('\n===== SCALED VERSIONS =====')
        scaled_X_boot_list = []
        transf_X_val_list = []
        transf_X_test_list = []

        for fold_id in range(1, NO_OF_FOLDS + 1):            
            X_boot_fold = self.X_boot_list[fold_id-1]
            X_val_fold = self.X_val_list[fold_id-1]

            scaled_X_boot_fold = []
            transf_X_val_fold = []
            transf_X_test_fold = []

            for run in range(NO_OF_BOOTSTRAPS):
                scaler = StandardScaler()

                scaled_X_boot = scaler.fit_transform(X_boot_fold[run])
                transf_X_val = scaler.transform(X_val_fold)
                transf_X_test = scaler.transform(self.X_test)
                
                scaled_X_boot_fold.append(scaled_X_boot)
                transf_X_val_fold.append(transf_X_val)
                transf_X_test_fold.append(transf_X_test)
                
                print(f'Fold {fold_id}/{NO_OF_FOLDS} - Getting Scaled/Transformed Versions.... {round(((run+1)/NO_OF_BOOTSTRAPS)*100)}%', end="\r")
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
        print('\n===== BOOTSTRAPPING =====')
        X_boot_list = []
        y_boot_list = []
        
        for fold_id in range(1, NO_OF_FOLDS+1):            
            X_train_split = self.X_train_split_list[fold_id-1]
            y_train_split = self.y_train_split_list[fold_id-1]
            
            X_boot_fold_list = []
            y_boot_fold_list = []
            for b in range(1, NO_OF_BOOTSTRAPS + 1):           
                print(
                    f"Fold {fold_id}/{NO_OF_FOLDS} - Getting bootstrapped samples: {round((b/NO_OF_BOOTSTRAPS)*100)}%",
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
    
    # 🗹 Initializes the class' variables through a provided dictionary
    def init_data_dict(self, data, force_reset=False):
        if force_reset:
            for key, value in data.items():
                setattr(self, key, value)
                
                if not force_reset:
                    continue
                jla.to_joblib(self.data_dir, key, value) 
            return
        
        for key in data:
            value = jla.get_data(self.data_dir, key)
            setattr(self, key, (
                np.asarray(value).ravel()
                if ('y_' in key or '_ids' in key) and not 'list' in key
                else value
            ))
    
    # =======================
    # MODEL-RELATED FUNCTIONS
    # =======================
    
    # 🗹 Gets the trained models as an array
    def train_models(self):
        print('===== TRAINING MODELS =====')
        
        # Random forest
        Random_Forest(
            self.X_boot_list,
            self.y_boot_list,
            NO_OF_FOLDS,
            NO_OF_BOOTSTRAPS,
            INSTABILITY_TYPE, 
            self.models_dir,
            FIXED_RANDOM_SEED
        )
        
        # XGBoost
        XGBoost(
            self.X_boot_list,
            self.y_boot_list,
            NO_OF_FOLDS,
            NO_OF_BOOTSTRAPS,
            INSTABILITY_TYPE, 
            self.models_dir,
            FIXED_RANDOM_SEED
        )
            
        # Logistic Regression
        Logistic_Regression(
            self.scaled_X_boot_list,
            self.y_boot_list,
            NO_OF_FOLDS,
            NO_OF_BOOTSTRAPS,
            INSTABILITY_TYPE, 
            self.models_dir,
            FIXED_RANDOM_SEED
        )
        
        # Linear SVC (SVM)
        Linear_SVC(
            self.scaled_X_boot_list,
            self.y_boot_list,
            NO_OF_FOLDS,
            NO_OF_BOOTSTRAPS,
            INSTABILITY_TYPE, 
            self.models_dir,
            FIXED_RANDOM_SEED
        )
        
        # Multilayer Perceptron
        Multilayer_Perceptron(
            self.scaled_X_boot_list,
            self.y_boot_list,
            NO_OF_FOLDS,
            NO_OF_BOOTSTRAPS,
            INSTABILITY_TYPE, 
            self.models_dir,
            FIXED_RANDOM_SEED
        )

    # 🗹 Gets the predictions from the trained models
    def generate_predictions(self):
        for eval_name in EVAL_NAMES:
            for model_name in MODEL_NAMES:
                print(f'Model: {model_name}')
                
                for fold_id in range(1, NO_OF_FOLDS+1):
                    if model_name in ['LR', 'LSVC', 'MLP']:
                        transf_X_eval_fold = (
                            self.transf_X_val_list[fold_id-1]
                            if eval_name == 'Validation'
                            else self.transf_X_test_list[fold_id-1]
                        )
                    else:
                        X_eval_fold = (
                            self.X_val_list[fold_id-1]
                            if eval_name == 'Validation'
                            else self.X_test
                        )
                    
                    for b in range(1, NO_OF_BOOTSTRAPS + 1):
                        print(f'Fold {fold_id}/{NO_OF_FOLDS} - Getting model predictions... {round((b/NO_OF_BOOTSTRAPS)*100)}%', end='\r')
                        
                        X_eval = (
                            transf_X_eval_fold[b-1]
                            if model_name in ['LR', 'LSVC', 'MLP']
                            else X_eval_fold
                        )

                        model = jla.get_model(self.models_dir, model_name, fold_id, b, 'Bootstrap')
                        y_pred_proba = model.predict_proba(X_eval)[:, 1]
                                            
                        csv_path = (f'{self.predictions_dir}/{model_name}_Predictions_{eval_name}_{fold_id}.csv')
                        csva.save_to_csv(csv_path, f'Bootstrap_{b}', y_pred_proba)
                        
                    print(f'Fold {fold_id}/{NO_OF_FOLDS} - Getting model predictions... 100%')
                print()
    
    # =========================
    # METRIC-SPECIFIC FUNCTIONS
    # ========================= 
      
    # 🗹 Gets all metric results  
    def generate_metrics(self, force_reset=False):
        print('\n===== METRICS =====')
        
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
        pm.calibration_curve()
        pm.supplementary_classification_metrics()
        pm.ninety_five_prediction_error()
        pm.mean_absolute_prediction_error()
        pm.classification_instability_index()
        pm.top_k_jaccard()
        pm.demographic_false_positive_rate()
        self.generate_mrip_results()
        
        self.generate_shap_values()
        pm.shap_analysis()  
    
    # 🗹 Gets the MRIP results       
    def generate_mrip_results(self):                
        for eval_name in EVAL_NAMES:
            for model_name in MODEL_NAMES:
                print(f'MRIP for {model_name}')
                
                mrip_dir = f'metrics/{INSTABILITY_TYPE}/MRIP/{model_name}'
                dir.create_nested(mrip_dir)

                records = []
                for fold_id in range(1, NO_OF_FOLDS + 1):
                    chosen_eval_ids = (
                        [self.val_ids_fold_list[fold_id-1][0]]
                        if eval_name == 'Validation'
                        else [self.test_ids[0]]
                    )
                    chosen_X_eval = (
                        self.X_val_list[fold_id-1].iloc[[0]]
                        if eval_name == 'Validation'
                        else self.X_test.iloc[[0]]
                    )
                    
                    for b in range(1, NO_OF_BOOTSTRAPS + 1):
                        print(
                            f'MRIP {model_name}: Fold {fold_id}/{NO_OF_FOLDS}, '
                            f'Bootstrap {b}/{NO_OF_BOOTSTRAPS}',
                            end='\r'
                        )
                        X_ref = self.X_boot_list[fold_id-1][b-1]
                        y_ref = self.y_boot_list[fold_id-1][b-1]
                        
                        model = jla.get_model(self.models_dir, model_name, fold_id, b, 'Bootstrap')

                        records.extend(rm.MRIP(
                            X_ref,
                            y_ref,
                            chosen_X_eval,
                            chosen_eval_ids,
                            model_name,
                            model,
                            fold_id,
                            b
                        ))
                    print()

                csv_path = f'{mrip_dir}/MRIP_{model_name}_{eval_name}.csv'
                rm.save_mrip_results(records, csv_path, run_label='Bootstrap')
    
    # 🗹 Generates the raw SHAP values for SHAP analysis
    def generate_shap_values(self):
        feature_names = self.X_train.columns
        
        for eval_name in EVAL_NAMES:
            for model_name in MODEL_NAMES:
                print(f'SHAP values for {model_name}')
                
                shap_dir = f'metrics/{INSTABILITY_TYPE}/shap_values/{model_name}'
                dir.create_nested(shap_dir)

                for fold_id in range(1, NO_OF_FOLDS + 1):
                    chosen_eval_ids = (
                        [self.val_ids_fold_list[fold_id-1][0]]
                        if eval_name == 'Validation'
                        else [self.test_ids[0]]
                    )
                    
                    shap_records = []
                    for b in range(1, NO_OF_BOOTSTRAPS + 1):
                        print(
                            f'SHAP {model_name}: Fold {fold_id}/{NO_OF_FOLDS}, '
                            f'Bootstrap {b}',
                            end='\r'
                        )
                        
                        if model_name in ['LR', 'LSVC', 'MLP']:
                            # Use scaled versions
                            X_ref = self.scaled_X_boot_list[fold_id-1][b-1]
                            chosen_X_eval = (
                                self.transf_X_val_list[fold_id-1][b-1][:1]
                                if eval_name == 'Validation' 
                                else self.transf_X_test_list[fold_id-1][b-1][:1]
                            )
                        else:
                            # Use raw versions
                            X_ref = self.X_boot_list[fold_id-1][b-1]
                            chosen_X_eval = (
                                self.X_val_list[fold_id-1].iloc[[0]]
                                if eval_name == 'Validation' 
                                else self.X_test.iloc[[0]]
                            )
                        
                        trained_model = jla.get_model(self.models_dir, model_name, fold_id, b, 'Bootstrap')
                        shap_records.append(rm.get_shap_values(
                            X_ref,
                            chosen_X_eval,
                            chosen_eval_ids,
                            feature_names,
                            model_name,
                            trained_model,
                            b,
                            run_label='Bootstrap'
                        ))

                    csv_path = f'{shap_dir}/SHAP_{model_name}_{eval_name}_Fold_{fold_id}.csv'
                    rm.save_shap_values(shap_records, csv_path)
