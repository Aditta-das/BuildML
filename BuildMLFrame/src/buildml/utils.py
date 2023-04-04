import pandas as pd
import numpy as np
from logger import logger
import joblib
import os


class FileCheck:
    def __init__(self):
        pass

    def check(self, name):
        try:
            if name.endswith(".xlsx"):
                df = pd.read_excel(name)
            elif name.endswith(".csv"):
                df = pd.read_csv(name)
            elif name.endswith(".feather"):
                df = pd.read_feather(name)
            start_mem_usg = df.memory_usage().sum() / 1024**2
            logger.info("Memory usage of properties dataframe is >> {:.2f} Mb".format(start_mem_usg))
            logger.info(f"Shape of dataset is: {df.shape}")
            prev_name = []
            new_name = []
            total_columns = len(df.columns.tolist())
            column_list = df.columns.tolist()
            for num in range(total_columns):
                if len(df[column_list[num]].name.split()) > 1:
                    name = df[column_list[num]].name.split()
                    joint = "_".join(name).lower()
                    prev_name.append(df[column_list[num]].name)
                    new_name.append(joint)

            # print(new_name)
            if len(new_name) > 0:
                logger.info("Change columns name")
                dictionary = {k: v for k, v in zip(prev_name, new_name)}
                df.rename(columns=dictionary, inplace=True)
            return df
            
        except Exception as e:
            logger.info("Files not recognised >> Convert it xlsx, csv, feather format")

    def reduce_memory_usage(self, df, verbose=True):
        # collected from internet
        numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
        start_mem = df.memory_usage().sum() / 1024 ** 2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024 ** 2
        if verbose:
            logger.info(
                "Memory usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                    end_mem, 100 * (start_mem - end_mem) / start_mem
                )
            )
        return df
