import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import metrics as m

def RF_Classifier(data):
    [
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        k, i,
        random_seed
    ] = data
    classifier = RandomForestClassifier(
        n_estimators=100, 
        random_state=random_seed
    )
    classifier.fit(X_train, y_train)
    
    X = [X_val, X_test] 
    y = [y_val, y_test]

    for idx in range(len(X)):
        # get the predictions using X_test and save it in y_pred
        y_pred_proba = classifier.predict_proba(X[idx])[:,1]

        predictions_df = pd.DataFrame({
            f'bootstrapped_{i}' : y_pred_proba
        })

        csv_file_path = f'predictions/RF_Predictions_{f'Validation' if idx == 0 else f'Test'}_{k}.csv'
        if i > 0:
            predictions_df = pd.read_csv(csv_file_path)
            predictions_df[f'bootstrapped_{i}'] = y_pred_proba
        predictions_df.to_csv(csv_file_path, index=False)

        m.generate_PR_AUC(y[idx], y_pred_proba, i, 'RF')
        
        #implement further metrics here 