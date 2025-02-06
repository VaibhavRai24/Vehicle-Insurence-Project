import sys
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object,read_yaml_file, save_numpy_array_data



class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path = SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)
        
        
    @staticmethod
    def read_data(file_path:str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
        
    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object for the data, 
        including gender mapping, dummy variable creation, column renaming,
        feature scaling, and type adjustments.
        
        """
        logging.info("Entered into the get_data_transformer_object method")
        try:
            numeric_transformer = StandardScaler()
            min_max_scaler = MinMaxScaler()
            logging.info("Transformer initialized : StandardScaler, MinMaxScaler")
            
            
            num_features = self._schema_config['num_features']
            mm_columns = self._schema_config["mm_columns"]
            logging.info("Numeric features and MinMax columns initialized")
            
            preprocessor = ColumnTransformer(
                transformers = [
                    ("StandardScaler", numeric_transformer, num_features),
                    ("MinMaxScaler", min_max_scaler, mm_columns)
                ],
                remainder = 'passthrough'
            )
            
            final_pipeline = Pipeline(steps = [ ("Preprocessor", preprocessor)])
            logging.info("Final Pipeline ready")
            logging.info("Exited from the get_data_transformer_object method")
            return final_pipeline
        
        except Exception as e:
            logging.exception("An exception occurred in get_data_transformer_object method")
            raise MyException(e, sys) from e
        
    def map_gender_column(self, df):
        """Map Gender column to 0 for Female and 1 for Male."""
        logging.info("Mapping Gender columns to binary values")
        df['Gender'] = df['Gender'].map({'Female': 0 , 'Male':1}).astype(int)
        return df
    
    def create_dummy_columns(self, df):
        """Create dummy variables for categorical features."""
        logging.info("Creating dummy variables for categorical features")
        df = pd.get_dummies(df, drop_first= True)
        return df
    
    def _rename_columns(self, df):
        """Rename specific columns and ensure integer types for dummy columns."""
        logging.info("Renaming specific columns and casting to int")
        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype('int')
        return df
        
        
    def drop_id_column(self,df):
        """Drop the 'id' column if it exists."""
        logging.info("Dropping the 'id column if it exists")
        drop_col = self._schema_config['drop_columns']
        if drop_col in df.columns:
            df = df.drop(drop_col, axis = 1)
        return df
            
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation initiated")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)
            
            train_df = self.read_data(file_path=self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train and Test data loaded Successfully")
            
            input_features_train_df = train_df.drop(columns = [TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            
            input_feature_test_df = test_df.drop(columns = [TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            
            logging.info("Input and Target cols defined for both train and test df.")
            
            input_features_train_df = self.map_gender_column(input_features_train_df)
            input_features_train_df = self.drop_id_column(input_features_train_df)
            input_features_train_df = self.create_dummy_columns(input_features_train_df)
            input_features_train_df = self._rename_columns(input_features_train_df)
            
            input_feature_test_df = self.map_gender_column(input_feature_test_df)
            input_feature_test_df = self.drop_id_column(input_feature_test_df)
            input_feature_test_df = self.create_dummy_columns(input_feature_test_df)
            input_feature_test_df = self._rename_columns(input_feature_test_df)
            
            logging.info("Transformation done end to end to train-test df.")
            
            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Data transformer object created")
            
            logging.info("Initializing transformation for train data")
            input_features_train_arr = preprocessor.fit_transform(input_features_train_df)
            logging.info("Initializing logging for the testing data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Transformation done for both train and test data")
            
            logging.info("Applying SMOTEENN for handling imbalanced dataset")
            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_features_train_arr, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )
            
            logging.info("SMOTEENN appiled to train_test df")
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info("feature-target concatenation done for train-test df.")
            
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array = train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array = test_arr)
            logging.info("Saving transformation object and transformed files")
            
            
            logging.info("Data Transformation Completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
            
        except Exception as e:
            raise MyException(e, sys) from e