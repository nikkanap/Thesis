import xgboost as xgb
import generate_prediction_csv

def XGBoost_Model(X, y, fold_id, random_seed, b=None):
    params = {
        'objective' : 'binary:logistic',
        'max_depth': 3,
        'learning_rate': 0.1,
        'random_state': random_seed
    }
    n = 50
    
    model = xgb.train(
        params=params,
        dtrain=X[0],
        num_boost_round=n,
    )
    
    instability_type = 'Stochastic' if b == None else 'Dataset'
    attribute_name = 'Random_Seed' if b == None else 'Bootstrap'
    idx = random_seed if b == None else b
    
    for X_idx in range(1, 3):
        test_type = 'Validation' if X_idx == 0 else 'Test'
        csv_file_path = f'predictions/{instability_type}/XGB_Predictions_{test_type}_{fold_id}.csv'
        
        # get the predictions and save it in y_pred_proba
        y_pred_proba = model.predict(X[X_idx])[:,1]
        
        # generate the predictions in a csv
        generate_prediction_csv(
            y_pred_proba,
            idx,
            csv_file_path,
            attribute_name
        )

        