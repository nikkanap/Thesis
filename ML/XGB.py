import pandas as pd
import metrics as m

import xgboost as xgb

def XGBoost_Model(data):
    [
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        k, i,
        random_seed
    ] = data
    
    xgb_train = xgb.DMatrix(X_train, y_train, enable_categorical=False)
    xgb_val = xgb.DMatrix(X_val, y_val, enable_categorical=False)
    xgb_test = xgb.DMatrix(X_test, y_test, enable_categorical=False)

    params = {
        'objective' : 'binary:logistic',
        'max_depth': 3,
        'learning_rate': 0.1,
        'random_state': random_seed
    }
    n = 50
    
    model = xgb.train(
        params=params,
        dtrain=xgb_train,
        num_boost_round=n,
    )
    
    X = [xgb_val, xgb_test] 
    y = [y_val, y_test]
    
    for idx in range(len(X)):
        # get the predictions using X_test and save it in y_pred
        y_pred_proba = model.predict(X[idx])

        predictions_df = pd.DataFrame({
            f'bootstrapped_{i}' : y_pred_proba
        })

        csv_file_path = f'predictions/XGB_Predictions_{f'Validation' if idx == 0 else f'Test'}_{k}.csv'
        if i > 0:
            predictions_df = pd.read_csv(csv_file_path)
            predictions_df[f'bootstrapped_{i}'] = y_pred_proba
        predictions_df.to_csv(csv_file_path, index=False)

        m.generate_PR_AUC(
            y[idx], 
            y_pred_proba, 
            i, 
            'XGB'
        )