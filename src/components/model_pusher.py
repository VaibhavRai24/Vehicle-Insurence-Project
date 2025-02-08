import sys

from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import Proj1Estimator

class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact, model_pusher_config:ModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        
        """
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.proj1_estimator = Proj1Estimator(bucket_name= model_pusher_config.bucket_name, model_path= model_pusher_config.s3_model_key_path)
        
        
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        logging.info("Entered initiate model pusher method of Model Trainer class")
        try:
            print("--------------------------------------------------->")
            logging.info("Uploading artifacts folder to s3 bucket ")
            self.proj1_estimator.save_model(from_file= self.model_evaluation_artifact.trained_model_path)
            model_pusher_arfifact = ModelPusherArtifact(bucket_name= self.model_pusher_config.bucket_name,
                                                        s3_model_path= self.model_pusher_config.s3_model_key_path)
            logging.info("Uploading artifacts folder to s3 bucket")
            logging.info(f"Model pusher artifact: [{model_pusher_arfifact}]")
            logging.info("Exited intiate model pusher method of modelTrainer")
            
            return model_pusher_arfifact
        
        except Exception as e:
            raise MyException("Model pusher failed") from e
               