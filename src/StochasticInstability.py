import numpy as np
import joblib

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
NO_OF_RANDOM_SEEDS = 10
INSTABILITY_TYPE = 'Stochastic'

# Constant lists
MODEL_NAMES = ['LR', 'LSVC', 'XGB', 'RF', 'MLP']
EVAL_NAMES = ['Validation', 'Test']

class StochasticInstability:
    def __init__(self, dataset=None, force_reset=False):   
        print('===== RUNNING STOCHASTIC INSTABILITY =====')   
        
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
        
        else:
            # Fetch only the existing data
            keys = ['X_train', 'y_train', 
                    'X_test', 'y_test', 
                    'train_ids', 'test_ids',
                    'X_train_split_list', 'y_train_split_list',
                    'X_val_list', 'y_val_list',
                    'val_ids_fold_list'
                ]
            self.init_data_dict(keys)

        [
            self.scaled_X_train_split_list,
            self.transf_X_val_list,
            self.transf_X_test_list
        ] = self.get_X_scaled()
        
        self.models_dir = f'models/{INSTABILITY_TYPE}'
        self.predictions_dir = f'predictions/{INSTABILITY_TYPE}'
        
        if force_reset:
            # Train the models
            self.train_models()
            dir.delete_nested(self.predictions_dir)
            dir.create_nested(self.predictions_dir)
            
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
            print(f'Splitting.... ({count}/{NO_OF_FOLDS})', end="\r")
            
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
        scaled_X_train_split_list = []
        transf_X_val_list = []
        transf_X_test_list = []

        for fold_id in range(1, NO_OF_FOLDS + 1):
            X_train_fold = self.X_train_split_list[fold_id-1]
            X_val_fold = self.X_val_list[fold_id-1]

            scaler = StandardScaler()

            scaled_X_train = scaler.fit_transform(X_train_fold)
            transf_X_val = scaler.transform(X_val_fold)
            transf_X_test = scaler.transform(self.X_test)
            
            scaled_X_train_split_list.append(scaled_X_train)
            transf_X_val_list.append(transf_X_val)
            transf_X_test_list.append(transf_X_test)

            print(f'Fold {fold_id}/{NO_OF_FOLDS}', end="\r")
        print(f'Fold {NO_OF_FOLDS}/{NO_OF_FOLDS}')
               
        return [
            scaled_X_train_split_list,
            transf_X_val_list,
            transf_X_test_list
        ]
    
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
        print('\n===== TRAINING MODELS =====')
        
        # Random forest
        Random_Forest(
            self.X_train_split_list,
            self.y_train_split_list,
            NO_OF_FOLDS,
            NO_OF_RANDOM_SEEDS,
            INSTABILITY_TYPE, 
            self.models_dir
        )
        
        # XGBoost
        XGBoost(
            self.X_train_split_list,
            self.y_train_split_list,
            NO_OF_FOLDS,
            NO_OF_RANDOM_SEEDS,
            INSTABILITY_TYPE,
            self.models_dir
        )
            
        # Logistic Regression
        Logistic_Regression(
            self.scaled_X_train_split_list,
            self.y_train_split_list,
            NO_OF_FOLDS,
            NO_OF_RANDOM_SEEDS,
            INSTABILITY_TYPE, 
            self.models_dir
        )
        
        # Linear SVC (SVM)
        Linear_SVC(
            self.scaled_X_train_split_list,
            self.y_train_split_list,
            NO_OF_FOLDS,
            NO_OF_RANDOM_SEEDS,
            INSTABILITY_TYPE, 
            self.models_dir
        )
        
        # Multilayer Perceptron
        Multilayer_Perceptron(
            self.scaled_X_train_split_list,
            self.y_train_split_list,
            NO_OF_FOLDS,
            NO_OF_RANDOM_SEEDS,
            INSTABILITY_TYPE, 
            self.models_dir
        )

    # 🗹 Gets the predictions from the trained models
    def generate_predictions(self):
        for eval_name in EVAL_NAMES:
            for model_name in MODEL_NAMES:
                print(f'Model: {model_name}')
                
                for fold_id in range(1, NO_OF_FOLDS + 1):
                    if model_name in ['LR', 'LSVC', 'MLP']:
                        X_eval = (
                            self.transf_X_val_list[fold_id-1]
                            if eval_name == 'Validation'
                            else self.transf_X_test_list[fold_id-1]
                        )
                    else:
                        X_eval = (
                            self.X_val_list[fold_id-1]
                            if eval_name == 'Validation'
                            else self.X_test
                        )

                    for random_seed in range(1, NO_OF_RANDOM_SEEDS + 1):
                        print(f'Fold {fold_id}/{NO_OF_FOLDS} - Getting predictions... {round((random_seed/NO_OF_RANDOM_SEEDS)*100)}%', end='\r')
                        
                        model = jla.get_model(self.models_dir, model_name, fold_id, random_seed, 'Random_Seed')
                        y_pred_proba = model.predict_proba(X_eval)[:, 1]

                        csv_path = (f'{self.predictions_dir}/{model_name}_Predictions_{eval_name}_{fold_id}.csv')
                        csva.save_to_csv(csv_path, f'Random_Seed_{random_seed}', y_pred_proba)
                    print(f'Fold {fold_id}/{NO_OF_FOLDS} - Getting predictions... 100%')
                print()
                  
    # ===============
    # METRIC-SPECIFIC
    # ===============      
    
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
        
        #pm.roc_auc()
        #pm.brier_score()
        #pm.calibration_plot()
        #pm.ninety_five_prediction_error()
        #pm.mean_absolute_prediction_error()
        #pm.classification_instability_index()
        #pm.top_k_jaccard()
        #pm.demographic_false_positive_rate()
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
                    X_ref = self.X_train_split_list[fold_id-1]
                    y_ref = self.y_train_split_list[fold_id-1]
                    
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

                    for random_seed in range(1, NO_OF_RANDOM_SEEDS + 1):
                        print(
                            f'MRIP {model_name}: Fold {fold_id}/{NO_OF_FOLDS}, '
                            f'Seed {random_seed}/{NO_OF_RANDOM_SEEDS}',
                            end='\r'
                        )
                        model = jla.get_model(self.models_dir, model_name, fold_id, random_seed, 'Random_Seed')

                        records.extend(rm.MRIP(
                            X_ref,
                            y_ref,
                            chosen_X_eval,
                            chosen_eval_ids,
                            model_name,
                            model,
                            fold_id,
                            random_seed
                        ))

                csv_path = f'{mrip_dir}/MRIP_{model_name}_{eval_name}.csv'
                rm.save_mrip_results(records, csv_path, run_label='Random Seed')
        
    # 🗹 Generates the raw SHAP values for SHAP analysis (FIX THIS)
    def generate_shap_values(self):
        feature_names = self.X_train.columns
        
        for eval_name in EVAL_NAMES:
            for model_name in MODEL_NAMES:
                print(f'SHAP for {model_name}')
                
                shap_dir = f'metrics/{INSTABILITY_TYPE}/shap_values/{model_name}'
                dir.create_nested(shap_dir)

                for fold_id in range(1, NO_OF_FOLDS + 1):
                    chosen_eval_ids = (
                        [self.val_ids_fold_list[fold_id-1][0]]
                        if eval_name == 'Validation'
                        else [self.test_ids[0]]
                    )
                    
                    if model_name in ['LR', 'LSVC', 'MLP']:
                        X_ref = self.scaled_X_train_split_list[fold_id-1]
                        chosen_X_eval = (
                            self.transf_X_val_list[fold_id-1][:1]
                            if eval_name == 'Validation'
                            else self.transf_X_test_list[fold_id-1][:1]
                        )
                    else:
                        X_ref = self.X_train_split_list[fold_id-1]
                        chosen_X_eval = (
                            self.X_val_list[fold_id-1].iloc[[0]]
                            if eval_name == 'Validation'
                            else self.X_test.iloc[[0]]
                        )
                        
                    shap_records = []
                    for random_seed in range(1, NO_OF_RANDOM_SEEDS + 1):
                        print(
                            f'SHAP {model_name}: Fold {fold_id}/{NO_OF_FOLDS}, '
                            f'Random seed {random_seed}/{NO_OF_RANDOM_SEEDS}',
                            end='\r'
                        )

                        trained_model = jla.get_model(self.models_dir, model_name, fold_id, random_seed, 'Random_Seed')
                        shap_records.append(rm.get_shap_values(
                            X_ref,
                            chosen_X_eval,
                            chosen_eval_ids,
                            feature_names,
                            model_name,
                            trained_model,
                            random_seed,
                            run_label='Random_Seed'
                        ))

                    csv_path = f'{shap_dir}/SHAP_{model_name}_{eval_name}_Fold_{fold_id}.csv'
                    rm.save_shap_values(shap_records, csv_path)

