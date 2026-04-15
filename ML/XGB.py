import pandas as pd
import metrics as m

import xgboost as xgb

def XGBoost_Model(X_train, y_train, X_test, y_test, i):
    xgb_train = xgb.DMatrix(X_train, y_train, enable_categorical=False)
    xgb_test = xgb.DMatrix(X_test, y_test, enable_categorical=False)

    params = {
        'objective' : 'binary:logistic',
        'max_depth': 3,
        'learning_rate': 0.1
    }
    n = 50
    
    model = xgb.train(
        params=params,
        dtrain=xgb_train,
        num_boost_round=n
    )

    # get the predictions using X_test and save it in y_pred
    y_pred_proba = model.predict(xgb_test)

    predictions_df = pd.DataFrame({
        'original_pred' if i == 0 else f'bootstrapped_{i}' : y_pred_proba
    })

    csv_file_path = 'predictions/XGB_Predictions.csv'
    if i > 0:
        predictions_df = pd.read_csv(csv_file_path)
        predictions_df[f'bootstrapped_{i}'] = y_pred_proba
    predictions_df.to_csv(csv_file_path, index=False)

    m.generate_PR_AUC(y_test, y_pred_proba, i, 'XGB')