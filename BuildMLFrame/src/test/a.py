import pandas as pd
import numpy as np

a = "/home/aditta/Desktop/BuildMLModel/BuildMLFrame/src/test/cars.csv"


df = pd.read_csv(a).copy()
# print(df.shape)
for col in df.columns.tolist():
#     miss = df[col].isnull().sum()
    if df[col].dtype == "bool":
        print(col)
#         get_idx = np.where(df[col].isnull() == True)[0]
#         df = df.drop(get_idx)
#         print(get_idx)
# print(df.shape)
# c = df.corr()
# print(c.index)
# print(df.dtypes)
