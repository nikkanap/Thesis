from sklearn.ensemble import RandomForestClassifier

from generate_prediction_csv import generate_prediction_csv
from runtime_metrics import MRIP

def RF_Classifier(predictions_dir, X_train, y_train, testing_data, fold_id, random_seed, b=None):
    print('Model: Random Forest')
    
    # Training the model
    classifier = RandomForestClassifier(
        n_estimators=100, 
        random_state=random_seed
    )
    classifier.fit(X_train, y_train)
    
    # Set up the generate_prediction_csv params
    nth_run = random_seed if b == None else b
    column_name = f'Random_Seed_{random_seed}' if b == None else f'Bootstrap_{b}'
       
    for test in testing_data:
        csv_file_path = f'{predictions_dir}/RF_Predictions_{test['name']}_{fold_id}.csv'
                
        # Get the predictions and save it in y_pred_proba
        y_pred_proba = classifier.predict_proba(test['X'])[:,1] 
        
        # Generate the predictions in a csv
        generate_prediction_csv(
            y_pred_proba,
            csv_file_path,
            column_name
        )
        
        # Generate an MRIP report for each defendant per nth_run
        MRIP(
            X_train=X_train,
            y_train=y_train,
            X_val_test=test['X'],
            y_val_test=test['y'],
            defendant_ids=test['ids'],
            trained_model=classifier,
            model_name='RF',
            test_name=test['name'],
            fold_id=fold_id,
            nth_run=nth_run
        )