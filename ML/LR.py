import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
import metrics as m

def LR_Model(X_train, y_train, X_test, y_test, i):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # get the predictions using X_test and save it in y_pred
    y_pred_proba = model.predict_proba(X_test)[:,1] #[:,1] is to get only positive values
    y_pred = model.predict(X_test)
    
    predictions_df = pd.DataFrame({
        'original_pred' if i == 0 else f'bootstrapped_{i}' : y_pred_proba
    })

    csv_file_path = 'predictions/LR_Predictions.csv'
    if i > 0:
        predictions_df = pd.read_csv(csv_file_path)
        predictions_df[f'bootstrapped_{i}'] = y_pred_proba
    predictions_df.to_csv(csv_file_path, index=False)

    m.generate_PR_AUC(y_test, y_pred_proba, i, 'LR')

    #implement further metrics here 
    