import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from dataclasses import dataclass

import warnings
warnings.filterwarnings('ignore')

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

            data=pd.read_csv('notebook/data/USA_Real_Estate.csv')

            logging.info("Read the dataset")

            artifact_dir=os.path.dirname(self.ingestion_congfig.train_data_path)
            os.makedirs(artifact_dir,exist_ok=True)

            logging.info('Removing unnecessary features and Feature Extraction')

            data.drop(columns=['prev_sold_date'],axis=1,inplace=True)

            logging.info('Reducing NAN Values')

            state_city_mode =data[data['city'].notna()].groupby('state')['city'].agg(lambda x: x.mode().iloc[0])  
            data['city'] = data.apply(lambda row: state_city_mode[row['state']] if pd.isna(row['city']) else row['city'],axis=1)

            brokered_city_mode =data[data['brokered_by'].notna()].groupby('city')['brokered_by'].agg(lambda x: x.mode().iloc[0])  
            data['brokered_by'] = data.apply(lambda row: brokered_city_mode.get(row['city'], row['brokered_by']) if pd.isna(row['brokered_by']) else row['brokered_by'],axis=1)

            street_city_mode =data[data['street'].notna()].groupby('city')['street'].agg(lambda x: x.mode().iloc[0])  
            data['street'] = data.apply(lambda row: street_city_mode.get(row['city'], row['street']) if pd.isna(row['street']) else row['street'],axis=1)

            zip_acre_median = data[data['acre_lot'].notna()].groupby('zip_code')['acre_lot'].median()
            state_acre_median = data[data['acre_lot'].notna()].groupby('state')['acre_lot'].median()

            data['acre_lot'] = data.apply(
                lambda row:
                    zip_acre_median.get(row['zip_code']) if pd.isna(row['acre_lot']) and row['zip_code'] in zip_acre_median
                    else state_acre_median.get(row['state']) if pd.isna(row['acre_lot']) and row['state'] in state_acre_median
                    else row['acre_lot'],
                axis=1
            )

            zip_house_median = data[data['house_size'].notna()].groupby('zip_code')['house_size'].median()
            street_house_median = data[data['house_size'].notna()].groupby('street')['house_size'].median()
            data['house_size'] = data.apply(
                lambda row:
                    zip_house_median.get(row['zip_code']) if pd.isna(row['house_size']) and row['zip_code'] in zip_house_median
                    else street_house_median.get(row['street']) if pd.isna(row['house_size']) and row['street'] in street_house_median
                    else row['house_size'],
                axis=1
            )

            house_bed_median = data[data['bed'].notna()].groupby('house_size')['bed'].median()
            acre_bed_median = data[data['bed'].notna()].groupby('acre_lot')['bed'].median()
            data['bed'] = data.apply(
                lambda row:
                    house_bed_median.get(row['house_size']) if pd.isna(row['bed']) and row['house_size'] in house_bed_median
                    else acre_bed_median.get(row['acre_lot']) if pd.isna(row['bed']) and row['acre_lot'] in acre_bed_median
                    else row['bed'],
                axis=1
            )

            house_bath_median = data[data['bath'].notna()].groupby('house_size')['bath'].median()
            acre_bath_median = data[data['bath'].notna()].groupby('acre_lot')['bath'].median()
            data['bath'] = data.apply(
                lambda row:
                    house_bath_median.get(row['house_size']) if pd.isna(row['bath']) and row['house_size'] in house_bath_median
                    else acre_bath_median.get(row['acre_lot']) if pd.isna(row['bath']) and row['acre_lot'] in acre_bath_median
                    else row['bath'],
                axis=1
            )

            data['price_per_sqft'] = data['price'] / data['house_size']
            data['price_per_room'] = data['price'] / (data['bed'] + data['bath'])

            data.dropna(inplace=True)

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

    transformation=DataTransformation()
    train_arr,test_arr,_=transformation.initiate_data_transformation(train_path=train_path,test_path=test_path)

    model=ModelTrainer()
    name,score,result=model.initiate_model_trainer(train_arr=train_arr,test_arr=test_arr)
    print(name,':',score,'\n\n',result)

