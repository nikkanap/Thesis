import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

import metrics as m

def LR_Model(data):
    [
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        k, i,
        random_seed
    ] = data
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_seed
    )
    model.fit(X_train, y_train)

    X = [X_val, X_test] 
    y = [y_val, y_test]

    for idx in range(len(X)):
        # get the predictions using X_test and save it in y_pred
        y_pred_proba = model.predict_proba(X[idx])[:,1] #[:,1] is to get only positive values
        #y_pred = model.predict(X[i])
        
        predictions_df = pd.DataFrame({
            f'bootstrapped_{i}' : y_pred_proba
        })

        csv_file_path = f'predictions/LR_Predictions_{f'Validation' if idx == 0 else f'Test'}_{k}.csv'
        if i > 0:
            predictions_df = pd.read_csv(csv_file_path)
            predictions_df[f'bootstrapped_{i}'] = y_pred_proba
        predictions_df.to_csv(csv_file_path, index=False)

        m.generate_PR_AUC(
            y[idx], 
            y_pred_proba, 
            i, 
            'LR'
        )

        #implement further metrics here 
    