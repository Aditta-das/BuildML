import os
import pandas as pd
import numpy as np
from utils import FileCheck
from logger import logger
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

class RunML():
    def __init__(self, path, label, output, testpath=None, task="classification", fillup="mean", seed=42, fold=5):
        self.path = path
        self.testpath = testpath
        self.label = label
        self.output = output
        self.task = task
        self.fillup = fillup
        self.seed = seed
        self.fold = fold
        self.le = LabelEncoder()
        self.lb = LabelBinarizer()

    def __post_init__(self):
        if os.path.exists(self.output):
            raise Exception("Output directory already exists. Please specify some other directory.")
        os.makedirs(self.output, exist_ok=True)
        logger.info(f"Output directory: {self.output}")

    def _build(self):

        #1. read file using pandas
        data = FileCheck()
        train_data = data.check(self.path)
        train_data = data.reduce_memory_usage(train_data)
        if self.testpath is not None:
            test_data = pd.read_csv(self.testpath)
            test_data = data.reduce_memory_usage(test_data)
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

                if missing_val > 0 and train_data[col].dtype != "object" or train_data[col].dtype == "bool" and col != self.label:
                    logger.info(f"Missing value of column name >> {col}: {missing_val}")
                    imputer = SimpleImputer(strategy=self.fillup, missing_values=np.nan)
                    imputer = imputer.fit(train_data[[col]])
                    train_data[col] = imputer.transform(train_data[[col]])
        

        #3. check is it classification or regression
        if self.task == "classification":
            #4. check label if classification problem. if object type convert to Label type. save file
            if label.dtype == "object" or label.dtype == "bool" and total_label > 2:
                logger.info(f"Label: {self.label} type object >> Converting in neumerical class")
                train_data[self.label] = self.le.fit_transform(train_data[self.label])

            elif label.dtype == "object" or label.dtype == "bool" and total_label == 2:
                logger.info(f"Label: {self.label} type object >> Converting in binary class")
                train_data[self.label] = self.lb.fit_transform(train_data[self.label])

            #4. check feature importance
            # Basic Feature remove: Constant feature. Ref: https://www.kaggle.com/code/raviprakash438/filter-method-feature-selection
            clf_label = train_data[self.label]
            train_data.drop(labels=[self.label], axis=1, inplace=True)
            
            varModel=VarianceThreshold(threshold=0) #Setting variance threshold to 0 which means features that have same value in all samples.
            varModel.fit(train_data)
            constArr=varModel.get_support()
            constCol=[col for col in train_data.columns if col not in train_data.columns[constArr]]
            logger.info(f"Constant columns name list: {constCol}" if len(constCol) > 0 else f"No constant columns.")
            train_data.drop(labels=constCol, axis=1, inplace=True)
            logger.info(f"Dropped constant colums" if len(constCol) > 0 else f"-------")

            #Create variance threshold model
            quasiModel=VarianceThreshold(threshold=0.01) #It will search for the features having 99% of same value in all samples.
            quasiModel.fit(train_data)
            quasiArr=quasiModel.get_support()
            quasiCols=[col for col in train_data.columns if col not in train_data.columns[quasiArr]]
            logger.info(f"Majority with same value columns name list: {quasiCols}" if len(quasiCols) > 0 else f"No major constant columns.")
            train_data.drop(columns=quasiCols, axis=1, inplace=True)
            logger.info(f"Dropped major constant colums" if len(quasiCols) > 0 else f"-------")
            print(train_data)

            #5. create kfold
            train_data[self.label] = clf_label
            train_data["kfold"] = -1
            train_data = train_data.sample(frac=1).reset_index(drop=True)
            kfold = StratifiedKFold(
                n_splits=self.fold,
                shuffle=True, 
                random_state=self.seed
            )
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X=train_data, y=train_data[f"{self.label}"].values)):
                train_data.loc[val_idx, "kfold"] = fold
        
        # fold and save fold
        for fold in range(self.fold):
            train_fold = train_data[train_data.kfold != fold].reset_index(drop=True)
            valid_fold = train_data[train_data.kfold == fold].reset_index(drop=True)

            train_fold.to_feather(os.path.join(self.output, f"train_fold_{fold}.feather"))
            valid_fold.to_feather(os.path.join(self.output, f"valid_fold_{fold}.feather"))
        

        

        #7. use hypertune optuna
        #8. Track using ml foundry

RunML(
    path="/home/aditta/Desktop/BuildMLModel/BuildMLFrame/src/test/cars.csv",
    label="is_exchangeable",
    output="/home/aditta/Desktop/BuildMLModel/BuildMLFrame/src/test/",
    task="classification",
    fillup="mean"
)._build()