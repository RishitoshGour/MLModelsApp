from sklearn.datasets import fetch_openml
import pandas as pd
from app import some_function

data = fetch_openml(name='breast-cancer', version=1, as_frame=True, parser='auto')
print("Columns:", data.frame.columns.tolist())
print("\nTarget column name:", data.target.name)
print("\nDataFrame shape:", data.frame.shape)
print("\nFirst row:")
print(data.frame.iloc[0])
print("\nData types:")
print(data.frame.dtypes)
