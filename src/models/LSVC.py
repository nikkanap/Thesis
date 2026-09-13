from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from generate_prediction_csv import generate_prediction_csv
from runtime_metrics import MRIP

def LinearSVC_Model(predictions_dir, X_train, y_train, testing_data, fold_id, random_seed, b=None):
    print('Model: Linear Support Vector Machine')
    
    # Training the model
    model = LinearSVC(
        random_state=random_seed
    )
    calibrated_svc = CalibratedClassifierCV(
        model,
        method='sigmoid',
        cv=3,
    )
    calibrated_svc.fit(X_train, y_train)
    model.fit(X_train, y_train)

    # Set up the generate_prediction_csv params
    nth_run = random_seed if b == None else b
    column_name = f'Random_Seed_{random_seed}' if b == None else f'Bootstrap_{b}'
        
    for test in testing_data:
        csv_file_path = f'{predictions_dir}/LSVC_Predictions_{test['name']}_{fold_id}.csv'
                        
        # Get the predictions and save it in y_pred_proba
        y_pred_proba = calibrated_svc.predict_proba(test['X'])[:,1] 
        
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
            trained_model=calibrated_svc,
            model_name='LSVC',
            test_name=test['name'],
            fold_id=fold_id,
            nth_run=nth_run
        )