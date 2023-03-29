import pandas as pd
import numpy as np
from checker import FileCheck
from logger import logger
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.impute import SimpleImputer

class RunML():
    def __init__(self, path, label, testpath=None, task="classification"):
        self.path = path
        self.testpath = testpath
        self.label = label
        self.task = task
        self.le = LabelEncoder()
        self.lb = LabelBinarizer()

    def _build(self):

        #1. read file using pandas
        data = FileCheck()
        train_data = data.check(self.path)
        # train_data = data.reduce_mem_usage(train_data)
        if self.testpath is not None:
            test_data = pd.read_csv(self.testpath)
        label = train_data[self.label]
        total_label = label.nunique() # count total label
        #2. check null columns if label is missing delete that row
        # get all name in list
        cols = train_data.columns.tolist()
        for col in cols:
            missing_val = train_data[col].isnull().sum()

            # percentage check
            percentage = (missing_val / train_data.shape[0])

            if col == self.label and missing_val > 0:
                get_idx = np.where(train_data[col].isnull() == True)[0]
                train_data = train_data.drop(get_idx).reset_index(drop=True)
                logger.info(f"New shape of dataset is: {train_data.shape}")
            
            # check missing value percentage above 60%
            if percentage >= 0.6 and col != self.label:
                logger.info(f"Null value percentage for column name >> {col}: is {percentage}")
                train_data.drop(col, inplace=True)
            else:
                # fill up columns
                if train_data[col].dtype == "object" or train_data[col].dtype == "bool" and col != self.label:
                    categorical_data = train_data[col].nunique()
                    if categorical_data > 2:
                        train_data[col] = self.le.fit_transform(train_data[col])
                    elif categorical_data == 2:
                        train_data[col] = self.lb.fit_transform(train_data[col])

        #3. check is it classification or regression
        if self.task == "classification":
            #4. check label if classification problem. if object type convert to Label type. save file
            if label.dtype == "object" or label.dtype == "bool" and total_label > 2:
                logger.info(f"Label: {self.label} type object >> Converting in neumerical class")
                
                train_data[self.label] = self.le.fit_transform(train_data[self.label])

            elif label.dtype == "object" or label.dtype == "bool" and total_label == 2:
                logger.info(f"Label: {self.label} type object >> Converting in binary class")
                
                train_data[self.label] = self.lb.fit_transform(train_data[self.label])

            # create kfold
            train_data["kfold"] = -1
            train_data = train_data.sample(frac=1).reset_index(drop=True)
            kfold = StratifiedKFold(
                n_splits=train_data[self.label].nunique(),
                shuffle=True, 
                random_state=42
            )
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X=train_data, y=train_data[f"{self.label}"].values)):
                train_data.loc[val_idx, "kfold"] = fold
        print(train_data)
        
        
        #5. check feature importance
        #6. make folds
        #7. use hypertune optuna
        #8. Track using ml foundry

RunML(
    path="/home/aditta/Desktop/BuildMLModel/BuildMLFrame/src/test/cars.csv",
    label="is_exchangeable",
    task="classification"
)._build()