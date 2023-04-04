import pandas as pd
import numpy as np

a = "/home/aditta/Desktop/BuildMLModel/BuildMLFrame/src/test/cars.csv"


import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv(a)
df["const"] = 1
df.to_csv("cars_const.csv", index=False)
print(df)
# create a sample dataset with missing values
# data = {'A': [1, 2, np.nan, 4],
#         'B': [5, np.nan, 7, 8],
#         'C': [9, 10, 11, np.nan]}
# df = pd.DataFrame(data)

# # create an instance of SimpleImputer with mean strategy
# imputer = SimpleImputer(strategy='mean')

# # fit the imputer on the data
# imputer.fit(df)

# # transform the data by replacing missing values with the mean
# imputed_data = imputer.transform(df)

# # convert the imputed data back to a pandas DataFrame
# imputed_df = pd.DataFrame(imputed_data, columns=df.columns)

# # print the imputed DataFrame
# print(imputed_df)

# import pandas as pd

# # create a sample DataFrame
# data = {'A': [1, 2, 3, 4],
#         'B': [5, 6, 7, 8],
#         'C': [9, 10, 11, 12]}
# df = pd.DataFrame(data)

# # create a Styler object
# styler = df.style

# # apply bold text to all values
# styler.set_properties(**{'font-weight': 'bold'})

# # display the styled DataFrame
# styler



# # df = pd.read_csv(a).copy()
# # print(df.shape)
# # for col in df.columns.tolist():
# #     miss = df[col].isnull().sum()
#     # if df[col].dtype == "bool":
#         # print(col)
# #         get_idx = np.where(df[col].isnull() == True)[0]
# #         df = df.drop(get_idx)
# #         print(get_idx)
# # print(df.shape)
# # c = df.corr()
# # print(c.index)
# # print(df.dtypes)
