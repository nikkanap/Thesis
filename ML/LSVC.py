from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from generate_prediction_csv import generate_prediction_csv

def LinearSVC_Model(X, y, X_val_ids, fold_id, random_seed, b=None):
    print('Model: Linear Support Vector Machine')
    model = LinearSVC(
        random_state=random_seed
    )
    calibrated_svc = CalibratedClassifierCV(
        model,
        method='sigmoid',
        cv=3,
    )
    calibrated_svc.fit(X[0], y[0])
    model.fit(X[0], y[0])

    instability_type = 'Stochastic' if b == None else 'Dataset'
    attribute_name = 'Random_Seed' if b == None else 'Bootstrap'
    idx = random_seed if b == None else b

    for X_idx in range(1, 3):
        test_type = 'Validation' if X_idx == 1 else 'Test'
        csv_file_path = f'predictions/{instability_type}/LinearSVC_Predictions_{test_type}_{fold_id}.csv'
         
        # get the predictions and save it in y_pred_proba
        y_pred_proba = calibrated_svc.predict_proba(X[X_idx])[:,1] 
        
        # generate the predictions in a csv
        generate_prediction_csv(
            y_pred_proba,
            X_val_ids,
            idx,
            csv_file_path,
            attribute_name,
        )