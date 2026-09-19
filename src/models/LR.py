
from sklearn.linear_model import LogisticRegression

def Logistic_Regression(
    X_train_arr,
    y_train_arr,
    no_of_folds,
    no_of_runs,
    instability_type, 
    random_seed=None
):
    print('Model: Logistic Regression')
    trained_model_arr = []
        
    for fold_id in range(1, no_of_folds + 1):
        X_train_fold = X_train_arr[fold_id-1]
        y_train_fold = y_train_arr[fold_id-1]
        
        trained_models_fold = []
        for nth_run in range(1, no_of_runs + 1):
            X_train = X_train_fold[nth_run-1]
            y_train = y_train_fold[nth_run-1]
            
            # Training the model
            model = LogisticRegression(
                max_iter=1000,
                random_state=(
                    random_seed 
                    if instability_type == 'Dataset'
                    else nth_run
                )
            )
            model.fit(X_train, y_train)
            trained_models_fold.append(model)
        trained_model_arr.append(trained_models_fold)
            
    return trained_model_arr

    """
    # Generate an MRIP report for each defendant per nth_run
    MRIP(
        test,
        instability_type,
        model, 'LR',
        test['name'], fold_id, nth_run
    )
    
    get_shap_values(
        X_train,
        X_val,
        test['X_raw'].columns,
        model,
        'LR',
        test['name'],
        fold_id,
        nth_run,
        instability_type,
        test['ids']
    )
    """
