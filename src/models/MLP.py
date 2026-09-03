from sklearn.neural_network import MLPClassifier

from generate_prediction_csv import generate_prediction_csv

def MLP_Model(X, y, X_val_ids, fold_id, random_seed, b=None):
    print('Model: Multilayer Perceptron')
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=1000,
        random_state=random_seed
    )
    model.fit(X[0], y[0])
    
    instability_type = 'Stochastic' if b == None else 'Dataset'
    attribute_name = 'Random_Seed' if b == None else 'Bootstrap'
    idx = random_seed if b == None else b

    for X_idx in range(1, 3):
        test_type = 'Validation' if X_idx == 1 else 'Test'
        csv_file_path = f'predictions/{instability_type}/MLP_Predictions_{test_type}_{fold_id}.csv'
            
        # get the predictions and save it in y_pred_proba
        y_pred_proba = model.predict_proba(X[X_idx])[:,1]

        # generate the predictions in a csv
        generate_prediction_csv(
            y_pred_proba,
            X_val_ids,
            idx,
            csv_file_path,
            attribute_name
        )