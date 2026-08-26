import pandas as pd
import matplotlib.pyplot as plt
import metrics as m

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

def LinearSVC_Model(data):
    [
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,        
        k, i,
        random_seed
    ] = data
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

    X = [X_val, X_test] 
    y = [y_val, y_test]
    
    for idx in range(len(X)):
        # get the predictions using X_test and save it in y_pred
        y_pred_proba = calibrated_svc.predict_proba(X[idx])[:,1] #[:,1] is to get only positive values
        #y_pred = model.predict(X_test)
        
        predictions_df = pd.DataFrame({
            f'bootstrapped_{i}' : y_pred_proba
        })

        csv_file_path = f'predictions/LSVC_Predictions_{f'Validation' if idx == 0 else f'Test'}_{k}.csv'
        if i > 0:
            predictions_df = pd.read_csv(csv_file_path)
            predictions_df[f'bootstrapped_{i}'] = y_pred_proba
        predictions_df.to_csv(csv_file_path, index=False)

        m.generate_PR_AUC(
            y[idx], 
            y_pred_proba, 
            i, 
            'LSVC'
        )

        #implement further metrics here 