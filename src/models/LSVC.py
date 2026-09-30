from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from create_dir import create_nested_directory
import joblib
import os

def Linear_SVC(
    X_train_arr,
    y_train_arr,
    no_of_folds,
    no_of_runs,
    instability_type, 
    models_dir,
    force_reset,
    random_seed=None                
):
    print('Model: Linear Support Vector Machine')
    final_model_dir = f'{models_dir}/LSVC'
    create_nested_directory(final_model_dir)
    
    for fold_id in range(1, no_of_folds + 1):
        X_train_fold = X_train_arr[fold_id-1]
        y_train_fold = y_train_arr[fold_id-1]
        for nth_run in range(1, no_of_runs + 1):
            print(f'Fold {fold_id}/{no_of_folds} ({round(((nth_run)/no_of_runs)*100)}%)', end='\r')
            model_path = os.path.join(
                final_model_dir,
                f"Fold_{fold_id}_{'Bootstrap' if instability_type == 'Dataset' else 'Random_Seed'}_{nth_run}.joblib"
            )
            if os.path.exists(model_path) and not force_reset:
                continue
            
            if instability_type == 'Dataset':
                X_train = X_train_fold[nth_run-1]
                y_train = y_train_fold[nth_run-1]
            else:
                X_train = X_train_fold
                y_train = y_train_fold
                
            # Training the model
            model = LinearSVC(
                random_state=(
                    random_seed 
                    if instability_type == 'Dataset'
                    else nth_run
                )
            )
            calibrated_svc = CalibratedClassifierCV(
                model,
                method='sigmoid',
                cv=3,
            )
            calibrated_svc.fit(X_train, y_train)   
            joblib.dump(calibrated_svc, model_path)
            
        print(f'Fold {fold_id}/{no_of_folds} (100%)')
    print()
        
