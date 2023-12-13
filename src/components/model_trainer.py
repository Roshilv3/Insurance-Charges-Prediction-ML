from dataclasses import dataclass

import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


@dataclass
class ModelTrainerConfig:
    model_trainer_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            model_report = []

            logging.info("Splitting the data into Train and Test")
            x_train, y_train, x_test, y_test = (
                train_arr[:,:-1],       # for x_train
                train_arr[:,-1],        # for y_train
                test_arr[:,:-1],        # for x_test
                test_arr[:,-1]          # for y_test
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()   
            }

            parameters = {
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[0.1,.01,0.05,0.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            model_report = evaluate_models(
                X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test,
                models=models,
                param=parameters
            )

            logging.info(f"Model Evaluation is completed")

            # To get the best scorer model from the dictionary.
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from the dictionary.
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found!")
            logging.info(f"Best model found in both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.model_trainer_file_path,
                obj = best_model
            )

            predicted = best_model.predict(x_test)

            R2_Score = r2_score(y_test, predicted)

            return ("{}:{}".format(best_model_name,R2_Score))
        
        except Exception as e:
            raise CustomException(e,sys)
     

