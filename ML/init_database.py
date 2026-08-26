from datasets import load_dataset
import pandas as pd

def init_dataset():
  # load dataset from HF
  dataset = load_dataset("imodels/compas-recidivism")
  training_df = pd.DataFrame(dataset['train'])  # training data
  testing_df = pd.DataFrame(dataset['test'])    # testing data

  # adding permanent identifier ids for metrics
  training_df["defendant_id"] = range(len(training_df))
  testing_df["defendant_id"] = range(len(testing_df))

  # separate training data to X_train and y_train
  X_train = training_df.drop(columns=['is_recid'])
  y_train = training_df['is_recid'].to_numpy()

  # same goes for the test data
  X_test = testing_df.drop(columns=['is_recid'])
  y_test = testing_df['is_recid'].to_numpy()
  
  return [X_train, y_train, X_test, y_test]