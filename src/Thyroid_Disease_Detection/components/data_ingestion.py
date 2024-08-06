import os
import sys
from src.Thyroid_Disease_Detection.exception import CustomException
from src.Thyroid_Disease_Detection.logger import logging
import pandas as pd
from src.Thyroid_Disease_Detection.utils import read_sql_data

from sklearn.model_selection import train_test_split

from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.inegstion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            #reading the data from mysql
            df=read_sql_data()
            logging.info("Reading completed from mysql database")

            os.makedirs(os.path.dirname(self.inegstion_config.train_data_path),exist_ok=True)

            df.to_csv(self.inegstion_config.raw_data_path,index=False,header=True)
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.inegstion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.inegstion_config.test_data_path,index=False,header=True)

            logging.info("Data ingestion is completed")

            return(
                self.inegstion_config.train_data_path,
                self.inegstion_config.test_data_path
            )
            


        except Exception as e:
            raise CustomException(e,sys)