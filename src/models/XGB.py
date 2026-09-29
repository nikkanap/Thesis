import xgboost as xgb

def XGBoost(
    X_train_arr,
    y_train_arr,
    no_of_folds,
    no_of_runs,
    instability_type, 
    random_seed=None
):
    print('Model: XGBoost')
    trained_model_arr = []
    
    for fold_id in range(1, no_of_folds + 1):
        print(f'Fold {fold_id}')
        X_train_fold = X_train_arr[fold_id-1]
        y_train_fold = y_train_arr[fold_id-1]
        
        trained_models_fold = []
        for nth_run in range(1, no_of_runs + 1):
            if instability_type == 'Dataset':
                X_train = X_train_fold[nth_run-1]
                y_train = y_train_fold[nth_run-1]
            else:
                X_train = X_train_fold
                y_train = y_train_fold
    
            # Setting up the parameters
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                max_depth=3,
                learning_rate=0.1,
                n_estimators=50,
                random_state=(
                    random_seed
                    if instability_type == 'Dataset'
                    else nth_run
                )
            )
            
            # Training the model
            model.fit(X_train, y_train)
            trained_models_fold.append(model)
        trained_model_arr.append(trained_models_fold)
            
    return trained_model_arr
                
                

        