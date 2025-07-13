import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_congfig=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Entered data ingestion method")

            data=pd.read_csv('data/Real_estate.csv')

            logging.info("Read the dataset")

            artifact_dir=os.path.dirname(self.ingestion_congfig.train_data_path)
            os.makedirs(artifact_dir,exist_ok=True)

            logging.info('Train and Test data separation initiated')

            train_data,test_data=train_test_split(data,test_size=0.2,random_state=42)

            logging.info('Spearated Train and Test data')

            train_data.to_csv(self.ingestion_congfig.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_congfig.test_data_path,index=False,header=True)

            logging.info('Data Ingestion Completed')

            return(
                self.ingestion_congfig.train_data_path,
                self.ingestion_congfig.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=='__main__':
    obj=DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()
