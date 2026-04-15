import pandas as pd
import metrics as m

from sklearn.neural_network import MLPClassifier

def MLP_Model(X_train, y_train, X_test, y_test, i):
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=1000,
        random_state=42
    )
    
    model.fit(X_train, y_train)

    # get the predictions using X_test and save it in y_pred
    y_pred_proba = model.predict_proba(X_test)[:,1]

    predictions_df = pd.DataFrame({
        'original_pred' if i == 0 else f'bootstrapped_{i}' : y_pred_proba
    })

    csv_file_path = 'predictions/MLP_Predictions.csv'
    if i > 0:
        predictions_df = pd.read_csv(csv_file_path)
        predictions_df[f'bootstrapped_{i}'] = y_pred_proba
    predictions_df.to_csv(csv_file_path, index=False)

    m.generate_PR_AUC(y_test, y_pred_proba, i, 'MLP')