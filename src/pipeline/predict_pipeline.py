import os
import sys
from src.utils import load_object
from src.exception import CustomException

import pandas as pd

class CustomData:
    def __init__(self,
                brokered_by: float,
                status: str,
                bed: float,
                bath: float,
                acre_lot: float,
                street: float,
                city: str,
                state: str,
                zip_code: float,
                house_size: float
                ):
        self.brokered_by=brokered_by
        self.status=status
        self.bed=bed
        self.bath=bath
        self.acre_lot=acre_lot
        self.street=street
        self.city=city
        self.state=state
        self.zip_code=zip_code
        self.house_size=house_size

    
    def get_data_as_data_frame(self):
        try:
            custom_input_data_dict={
                "brokered_by":[self.brokered_by],
                "status":[self.status],
                "bed":[self.bed],
                "bath":[self.bath],
                "acre_lot":[self.acre_lot],
                "street":[self.street],
                "city":[self.city],
                "state":[self.state],
                "zip_code":[self.zip_code],
                "house_size":[self.house_size],

            }
            
            return pd.DataFrame(custom_input_data_dict)

        except Exception as e:
            raise CustomException(e,sys)
        

class predictPipeline:
    def __init__(self):
        pass

    def predict_result(self,features):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            processed_data=preprocessor.transform(features)
            result=model.predict(processed_data)

            return result

        except Exception as e:
            raise CustomException(e,sys)
