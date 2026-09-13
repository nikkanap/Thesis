import pandas as pd

def generate_prediction_csv(
  y_pred_proba,
  column_name,
  csv_file_path
):
  df = pd.DataFrame({ column_name : y_pred_proba })
  df.to_csv(csv_file_path, index=False)