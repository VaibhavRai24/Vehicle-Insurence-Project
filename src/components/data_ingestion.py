import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataInegestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.data_access.proj1_data import Proj1Data

class DataIngestion:
    def __init__(self, data_ingestion_config: DataInegestionConfig = DataInegestionConfig()):
        """:param data_ingestion_config: configuration for data ingestion
        """
        try: 
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e, sys)
    
    def export_data_into_feature_store(self) -> DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from mongodb to csv file
        
        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Exporting data into feature store")
            my_data = Proj1Data()
            dataframe = my_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            logging.info(f"Dataframe shape: {dataframe.shape}")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Exporting data to {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise MyException(e, sys)
        
    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio 
        
        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Splitting data into train and test set")
        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio, random_state=42)
            logging.info("Performed the train test split on the dataframe")
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Exporting train and test set files")
            
            train_set.to_csv(self.data_ingestion_config.training_file_path, index = False, header = True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index = False, header = True)
            logging.info(f"Exported train and test set files")
            
        except Exception as e:
            raise MyException(e, sys) from e
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion process
        
        Output      :   DataIngestionArtifact object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            dataframe = self.export_data_into_feature_store()
            logging.info("Got the data from the mongodb")
            self.split_data_as_train_test(dataframe)
            logging.info("Performed the train test split on the dataset")
            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )
            data_ingestion_artifact = DataIngestionArtifact(train_file_path=self.data_ingestion_config.training_file_path, test_file_path=self.data_ingestion_config.testing_file_path)
            logging.info("DataIngestionArtifact object created")
            return data_ingestion_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e
            