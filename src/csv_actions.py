import pandas as pd
import numpy as np
import os  
  
# Saves data to csv files
def save_to_csv(
    csv_path,
    column_name=None,
    data=None,
):
    if isinstance(data, (dict, pd.DataFrame)):
        pd.DataFrame(data).to_csv(csv_path, index=False)
        return

    values = list(data)
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        df[column_name] = values
    else:
        df = pd.DataFrame({ column_name: values })
    df.to_csv(csv_path, index=False)