import os
import sys
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model

from sklearn import tree
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_path_obj=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):

        try:
            logging.info('Separating X_train,X_test,y_train,y_test')

            X_train,X_test,y_train,y_test=(
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]
            )

            models={
                "Decision Tree":tree.DecisionTreeRegressor(),
                "LightGBM":LGBMRegressor(),
                "XGBoost":XGBRegressor()
            }

            param_grid = {
                    "Decision Tree": {
                        "max_depth": [10],
                        "min_samples_split": [5],
                        "min_samples_leaf": [3],
                        "random_state": [42]
                    },
                    "LightGBM": {
                        "n_estimators": [500],
                        "learning_rate": [0.05],
                        "max_depth": [8],
                        "num_leaves": [31],
                        "random_state": [42]
                    },
                    "XGBoost": {
                        "n_estimators": [500],
                        "learning_rate": [0.05],
                        "max_depth": [8],
                        "subsample": [0.8],
                        "colsample_bytree": [0.8],
                        "random_state": [42]
                    }
            }


            logging.info('Sending models to evaluate')

            y_train = np.log1p(y_train)
            y_test = np.log1p(y_test)


            result=evaluate_model(X_train,X_test,y_train,y_test,models,param_grid)

            logging.info("Recieved result from model evaluation")

            best_score=max(result.values())
            best_model_name=list(models.keys())[list(result.values()).index(best_score)]
            best_model=models[best_model_name]
            logging.info("Found best model")

            
            logging.info('Saving model.pkl')

            save_object(
                file_path=self.model_trainer_path_obj.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            score=r2_score(y_test,predicted)
            return best_model_name,score,result.values()

        except Exception as e:
            raise CustomException(e,sys)
        
        

