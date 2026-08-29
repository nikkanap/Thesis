import pandas as pd
import operator as op
import os

def generate_prediction_csv(
  y_pred_proba,
  X_val_ids,
  idx,
  csv_file_path,
  attribute_name
):
  if not os.path.isfile(csv_file_path):
    val_id_df = pd.DataFrame({
      f'defendant_id': X_val_ids if op.contains(csv_file_path, 'Validation') else range(len(y_pred_proba))
    })
    val_id_df.to_csv(csv_file_path, index=False)
    
  predictions_df = pd.DataFrame({
    f'{attribute_name}_{idx}' : y_pred_proba
  })

  predictions_df = pd.read_csv(csv_file_path)
  predictions_df[f'{attribute_name}_{idx}'] = y_pred_proba
  predictions_df.to_csv(csv_file_path, index=False)