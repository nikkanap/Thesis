import pandas as pd

def generate_prediction_csv(
  y_pred_proba,
  idx,
  csv_file_path,
  attribute_name
):
  predictions_df = pd.DataFrame({f'{attribute_name}_{idx}' : y_pred_proba})

  if idx > 0:
      predictions_df = pd.read_csv(csv_file_path)
      predictions_df[f'{attribute_name}_{idx}'] = y_pred_proba
  predictions_df.to_csv(csv_file_path, index=False)