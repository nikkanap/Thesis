from sklearn.ensemble import RandomForestClassifier

from generate_prediction_csv import generate_prediction_csv
from runtime_metrics import calculate_mrip

def RF_Classifier(predictions_dir, X_train, y_train, X_val, y_val, X_test, y_test, fold_id, random_seed, b=None):
    print('Model: Random Forest')
    classifier = RandomForestClassifier(
        n_estimators=100, 
        random_state=random_seed
    )
    classifier.fit(X_train, y_train)
    
    testing_data = [
        {
            'name': 'Validation',
            'X': X_val,
            'y': y_val,
        }, 
        {
            'name': 'Test',
            'X': X_test,
            'y': y_test,
        }, 
    ]
    
    for test in testing_data:
        csv_file_path = f'{predictions_dir}/RF_Predictions_{test['name']}_{fold_id}.csv'
                
        # get the predictions and save it in y_pred_proba
        y_pred_proba = classifier.predict_proba(test['X'])[:,1] 
        
        column_name = f'Random_Seed_{random_seed}' if b == None else f'Bootstrap_{b}'
        
        # generate the predictions in a csv
        generate_prediction_csv(
            y_pred_proba,
            csv_file_path,
            column_name
        )
        
        # also generate an MRIP report for each defendant per nth_run
        calculate_mrip(
            X=test['X'],
            y_true=test['y'],
            trained_model=classifier,
            model_name='RF',
            instability_type=instability_type,
            test_name=test['name'],
            fold_id=fold_id,
            nth_run=random_seed if b == None else b
        )