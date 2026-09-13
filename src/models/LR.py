
from sklearn.linear_model import LogisticRegression

from generate_prediction_csv import generate_prediction_csv
from runtime_metrics import MRIP

def LR_Model(predictions_dir, X_train, y_train, testing_data, fold_id, random_seed, b=None):
    print('Model: Logistic Regression')
    
    # Training the model
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_seed
    )
    model.fit(X_train, y_train)
    
    # Set up the generate_prediction_csv params
    nth_run = random_seed if b == None else b
    column_name = f'Random_Seed_{random_seed}' if b == None else f'Bootstrap_{b}'
            
    for test in testing_data:
        csv_file_path = f'{predictions_dir}/LR_Predictions_{test['name']}_{fold_id}.csv'
        
        # Get the predictions and save it in y_pred_proba
        y_pred_proba = model.predict_proba(test['X'])[:,1] 
        
        # Generate the predictions in a csv
        generate_prediction_csv(
            y_pred_proba,
            csv_file_path,
            column_name,
        )
        
        # Generate an MRIP report for each defendant per nth_run
        MRIP(
            X_train=X_train,
            y_train=y_train,
            X_val_test=test['X'],
            y_val_test=test['y'],
            defendant_ids=test['ids'],
            trained_model=model,
            model_name='LR',
            test_name=test['name'],
            fold_id=fold_id,
            nth_run=nth_run
        )
    