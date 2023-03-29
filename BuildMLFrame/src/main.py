import pandas as pd
import argparse
from file_checker import FileCheck
from logger import logger
from sklearn.preprocessing import LabelEncoder

class RunML:
    def __init__(self):
        self.path = args.path
        self.testpath = args.testpath
        self.label = args.label
        self.encoder = args.encoder

    def run(self):
        
        # read data
        data = FileCheck()
        train_data = data.check(self.path)
        if self.testpath is not None:
            test_data = pd.read_csv(self.testpath)
        label = train_data[self.label]

        # Rename columns 
        # convert column name
        new_name = []
        total_columns = len(train_data.columns.tolist())
        column_list = train_data.columns.tolist()
        for num in range(total_columns):
            if len(train_data[column_list[num]].name.split()) > 1:
                name = train_data[column_list[num]].name.split()
                joint = "_".join(name).lower()
                new_name.append(joint)
        
        # save dictionary as file
        # check if new_name length is > 0
        if len(new_name) > 0:
            logger.info("Change columns name")
            # save column names
            dictionary = {k: v for k, v in zip(column_list, new_name)}
            train_data.rename(columns=dictionary, errors="raise")
            data.save_name_converter(dictionary)
            logger.info("File Saved >> storage/name_chager.txt")
        
        # Check any null values
        for col in train_data.columns.tolist():
            if train_data[col].isnull().sum() > 0:
                logger.info("Null columns: {train_data[col]}")
        logger.info("No null values found")

        # convert Label in numeric if it is object type
        if label.dtype == "object":
            logger.info(f"Label: {self.label} type object >> Converting in neumerical")
            le = LabelEncoder()
            train_data[self.label] = le.fit_transform(train_data[self.label])
        
        # check others columns is object type or not
        for col in train_data.columns.tolist():
            if train_data[col].dtype == "object":
                pass
            
            # Save file encoder
            _ = FileCheck().save_my_file(le, file_name=f"{self.encoder}.joblib")
            logger.info(f"File Saved >> storage/{self.encoder}.joblib")
        
        print(train_data.columns)


        
        
        

        
        
        
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", 
        type=str, 
        default=None, 
        help="add your data path. i.e: /usr/path/hello.csv",
        required=True
    )
    parser.add_argument(
        "--testpath",
        type=str,
        required=False,
        help="add your test data path. i.e: /usr/path/hello.csv",
    )
    parser.add_argument(
        '--label', 
        type=str, 
        required=True,
        default=None, 
        help='set your label name'
    )
    parser.add_argument(
        '--encoder', 
        type=str, 
        required=False, 
        default="encoder",
        help='encoder name. i.e: new.joblib'
    )
    

    args = parser.parse_args()

    RunML().run()