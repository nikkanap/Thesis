import pandas as pd
import os

def generate_prediction_csv(
  y_pred_proba,
  column_name,
  csv_file_path
):
  if os.path.isfile(csv_file_path):
      df = pd.read_csv(csv_file_path)
      df[column_name] = y_pred_proba
  else:
      df = pd.DataFrame({ column_name : y_pred_proba })
  df.to_csv(csv_file_path, index=False)