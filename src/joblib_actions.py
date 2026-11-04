
import os
import joblib

# Checks if the joblib data exists
def data_exists(data_dir, data_name):
    csv_path = f'{data_dir}/{data_name}.joblib'
    return os.path.isfile(csv_path)

# Gets the data from the joblib file
def get_data(data_dir, data_name):
    return joblib.load(f'{data_dir}/{data_name}.joblib')

# Turns the data to a joblib file
def to_joblib(data_dir, data_name, data):
    print(f'Saving data: {data_name}')
    joblib.dump(data,f'{data_dir}/{data_name}.joblib')

# Gets a trained model from its joblib file
def get_model(model_dir, model_name, fold_id, nth_run, run_name):
    return joblib.load(f"{model_dir}/{model_name}/Fold_{fold_id}_{run_name}_{nth_run}.joblib")