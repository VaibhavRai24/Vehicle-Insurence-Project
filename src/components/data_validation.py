import json
import os
import sys
import pandas as pd
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config:DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)
        
    def validate_number_of_columns(self, dataframe:DataFrame) -> bool:
        """
        Method Name :   validate_number_of_columns
        Description :   This method validates the number of columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required colums present : [{status}]")
            return status
        except Exception as e:
            raise MyException(e, sys)
        
    def is_coloumn_exist(self, df:DataFrame) -> bool:
        """
        Method Name :   is_column_exist
        Description :   This method validates the existence of a numerical and categorical columns
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)
            if len(missing_numerical_columns)>0:
                logging.error(f"Missing numerical columns : {missing_numerical_columns}")
                
            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)
                    
            if len(missing_categorical_columns)>0:
                logging.error(f"Missing categorical columns : {missing_categorical_columns}")
                
            return False if len(missing_numerical_columns)>0 or len(missing_categorical_columns)>0 else True
        
        except Exception as e:
            raise MyException(e, sys) from e
        
        
    @staticmethod
    def read_data(file_path:str) -> DataFrame:
        """
        Method Name :   read_data
        Description :   This method reads the data from the file path
        
        Output      :   Returns DataFrame
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
        
    def initialize_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            validation_error_msg = ""
            logging.info("Starting the data validation")
            train_df, test_df = (DataValidation.read_data(file_path = self.data_ingestion_artifact.train_file_path),
                                 DataValidation.read_data(file_path = self.data_ingestion_artifact.test_file_path))
            
            status = self.validate_number_of_columns(dataframe = train_df)
            if not status:
                validation_error_msg += "Number of columns mismatch in train data\n"
            else:
                logging.info("All the required columns are present in the train dataframe: {status}")
                
            status = self.validate_number_of_columns(dataframe= test_df)
            if not status:
                validation_error_msg += "Number of columns mismatch in test data\n"
            else:
                logging.info(" All the required columns are present in the test dataframe: {status}")

            status = self.is_coloumn_exist(df=train_df)
            if not status:
                validation_error_msg += "Columns are missing in the train dataframe"
            else:
                logging.info(" All the required columns are present in the train dataframe: {status}")
                
            status = self.is_coloumn_exist(df=test_df)
            if not status:
                validation_error_msg += "Columns are missing in the test dataframe"
            else:
                logging.info("All the required columns are present in the test dataframe: {status}")
                
            validation_status  = len(validation_error_msg) ==0
            
            data_validation_artifact = DataValidationArtifact(validation_status=validation_status,
                                                              message=validation_error_msg,
                                                              validation_report_file_path=self.data_validation_config.validation_report_file_path)       
            
            report_dir = os.path.dirname(self.data_validation_config.validation_report_file_path)
            os.makedirs(report_dir, exist_ok= True)
            
            validation_report = {
                "validation_status": validation_status,
                "validation_error_msg": validation_error_msg.strip()
            }
            
            with open(self.data_validation_config.validation_report_file_path, 'w') as report_file:
                json.dump(validation_report, report_file, indent=4)
                
            logging.info("Data validation artifact created and saved to JSON file")
            logging.info("Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e