import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.preprocessor_obj_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            num_col=['brokered_by']
            cat_col=['status','city','state']

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median'))
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinal_encoder',OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                ]
            )

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_col),
                    ('cat_pipeline',cat_pipeline,cat_col)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read the train and test part')

            preprocessing_obj=self.get_data_transformer_obj()
            logging.info('Obtained preprocessing object')

            target_feature_name='price'

            input_feature_train=train_df.drop(columns=[target_feature_name],axis=1)
            input_feature_test=test_df.drop(columns=[target_feature_name],axis=1)

            target_feature_train=train_df[target_feature_name]
            target_feature_test=test_df[target_feature_name]

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test)

            train_arr = np.concatenate([input_feature_train_arr, np.array(target_feature_train).reshape(-1, 1)],axis=1)
            test_arr = np.concatenate([input_feature_test_arr, np.array(target_feature_test).reshape(-1, 1)],axis=1)

            logging.info('Preprocessing is complete')

            save_object(
                file_path=self.preprocessor_obj_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Created preprocessor.pkl')

            return (
                train_arr,
                test_arr,
                self.preprocessor_obj_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
