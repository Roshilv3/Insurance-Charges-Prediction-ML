import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        ## This function is responsible for data transformation for different features.
        try:
            num_col = ["age","bmi","children"]
            cat_col = ["sex","smoker","region"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one-hot-encoder",OneHotEncoder()),
                    ("scaler",StandardScaler())
                ]
            )

            logging.info(f"Categorical column: {cat_col}")
            logging.info(f"Numerical column: {num_col}")

            preprocessor = ColumnTransformer(
                [
                ("num_pipeline", num_pipeline,num_col),
                ("cat_pipeline", cat_pipeline,cat_col)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test dataset")
            
            # Obtaining preprocessing object.
            preprocessing_obj = self.get_data_transformation_obj()

            target_col = "charges"
            numeric_col = ["age","bmi","children"]

            input_feature_train_df = train_df.drop(columns=[target_col],axis=1)
            target_feature_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(columns=[target_col],axis=1)
            target_feature_test_df = test_df[target_col]

            logging.info(f"Applying preprocessing object on training and testing data")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Save Preprocessing Object")

            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.get_data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
