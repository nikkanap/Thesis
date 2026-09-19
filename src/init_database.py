from datasets import load_dataset
import pandas as pd

def init_dataset():
    # load dataset from HF
    dataset = load_dataset("imodels/compas-recidivism")
    training_df = pd.DataFrame(dataset['train'])
    testing_df = pd.DataFrame(dataset['test'])

    # adding permanent identifier ids for metrics
    training_df["defendant_id"] = range(len(training_df))
    testing_df["defendant_id"] = range(len(testing_df))
    
    # training data (no ids)
    X_train = training_df.drop(columns=['is_recid', 'defendant_id'])
    y_train = training_df['is_recid'].to_numpy()

    # test data (no ids)
    X_test = testing_df.drop(columns=['is_recid', 'defendant_id'])
    y_test = testing_df['is_recid'].to_numpy()
    
    # training and test ids
    training_ids = training_df['defendant_id'].to_numpy()
    testing_ids = testing_df['defendant_id'].to_numpy()

    return [X_train, y_train, X_test, y_test, training_ids, testing_ids]