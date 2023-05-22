import os, sys
import pandas as pd
import numpy as np
from visa.constant import *
from visa.logger import logging
from visa.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig
from visa.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, ModelPusherArtifact
from visa.exception import CustomException
from datetime import date
from collections import namedtuple
from visa.config.configuration import Configuration
from visa.components.data_ingestion import DataIngestion
from visa.components.data_validation import DataValidation
from visa.components.data_transformation import DataTransformation
from visa.components.model_trainer import ModelTrainer
from visa.components.model_evaluation import ModelEvaluation
from visa.components.model_pusher import ModelPusher

import uuid
from threading import Thread
from typing import List, Dict 
from multiprocessing import Process
from visa.constant import EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME

Experiment = namedtuple("Experiment", ["experiment_id", "initialization_timestamp", "artifact_time_stamp",
                                      "running_status", "start_time", "stop_time", "excution_time", "message",
                                      "experiment_file_path", "accuracy", "is_model_accepted"])

class TrainingPipeline(Thread):
    experiment: Experiment = Experiment(*([None] * 11))
    experiment_file_path = None

    def __init__(self, config: Configuration = Configuration())->None:
        try:
            os.makedirs(config.training_pipeline_config.artifact_dir, exist_ok=True)
            TrainingPipeline.experiment_file_path = os.path.join(config.training_pipeline_config.artifact_dir, EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME)
            super().__init__(daemon=False, name="pipeline")
            self.config = config

        except Exception as e:
            raise CustomException(e,sys) from e  
        
    def start_data_ingestion(self)-> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config= self.config.get_data_ingestion_config())
            return data_ingestion.initate_data_ingestion()
        except Exception as e:
            raise CustomException(e,sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                             data_ingestion_artifact=data_ingestion_artifact)
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise CustomException(e,sys) from e   
        
    def start_data_transformation(self, 
                                  data_ingestion_artifact: DataIngestionArtifact,
                                  data_validation_artifact: DataValidationArtifact
                                  ) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(
                data_transformation_config=self.config.get_data_transformation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise CustomException(e,sys) from e 

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(
                model_trainer_config=self.config.get_model_trainer_config(),
                data_transformation_artifact=data_transformation_artifact
            )
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise CustomException(e,sys) from e 

    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact,
                               data_validation_artifact: DataValidationArtifact,
                               model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        try:
            model_eval = ModelEvaluation(
                model_evaluation_config=self.config.get_model_evaluation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                model_trainer_artifact=model_trainer_artifact)
            return model_eval.initiate_model_evaluation()
        except Exception as e:
            raise CustomException(e, sys) from e 

    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        try:
            model_pusher = ModelPusher(
                model_pusher_config=self.config.get_model_pusher_config(),
                model_evaluation_artifact=model_evaluation_artifact
            )
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            raise CustomException(e, sys) from e 
        
    def run_pipeline(self):
        try:
            if TrainingPipeline.experiment.running_status:
                logging.info("Training Pipeline is already running.")
                return TrainingPipeline.experiment

            logging.info("Training Pipeline is starting.")

            experiment_id = str(uuid.uuid4())

            TrainingPipeline.experiment = Experiment(experiment_id=experiment_id,
                                                    initialization_timestamp=self.config.time_stamp,
                                                    artifact_time_stamp=self.config.time_stamp,
                                                    running_status=True,
                                                    start_time=datetime.now(),
                                                    stop_time=None,
                                                    excution_time=None,
                                                    experiment_file_path=TrainingPipeline.experiment_file_path,
                                                    is_model_accepted=None,
                                                    message="Training pipeline has been started.",
                                                    accuracy=None,
                                                    )

            logging.info(f"Training Pipeline experiment: {TrainingPipeline.experiment}")

            self.save_experiment()
            
            # Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()
            
            # Data Validation
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact = data_ingestion_artifact)
            
            # Data Transformation
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact = data_ingestion_artifact,
                data_validation_artifact = data_validation_artifact
            )

            # Model Trainer
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )

            # Model Evaluation 
            model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
                                                                    data_validation_artifact=data_validation_artifact,
                                                                    model_trainer_artifact=model_trainer_artifact)

            # Model Pusher
            if model_evaluation_artifact.is_model_accepted:
                model_pusher_artifact = self.start_model_pusher(
                    model_evaluation_artifact=model_evaluation_artifact
                )
                logging.info(f"Model Pusher Artifact: {model_pusher_artifact}")
            else:
                logging.info(f"Trained model rejected.")
            logging.info(f"Training Pipeline Completed.")

            stop_time = datetime.now() 
            TrainingPipeline.experiment = Experiment(experiment_id=TrainingPipeline.experiment_id,
                                                     initialization_timestamp=self.config.time_stamp,
                                                     artifact_time_stamp=self.config.time_stamp,
                                                     running_status=False,
                                                     start_time=Pipeline.experiment.start_time,
                                                     stop_time=stop_time,
                                                     excution_time=stop_time - Pipeline.experiment.start_time,
                                                     message="Pipeline has been completed.",
                                                     experiment_file_path=Pipeline.experiment_file_path,
                                                     is_model_accepted=model_evaluation_artifact.is_model_accepted,
                                                     accuracy=model_trainer_artifact.model_accuracy
                                                     )
            logging.info(f"Training Pipeline experiment: {TrainingPipeline.experiment}")
            self.save_experiment()
        except Exception as e:
            raise CustomException(e,sys) from e  

    def run(self):
        try:
            self.run_pipeline()
        except Exception as e:
            raise e 

    def save_experiment(self):
        try:
            if TrainingPipeline.experiment.experiment_id is not None:
                experiment = TrainingPipeline.experiment
                experiment_dict = experiment._asdict()
                experiment_dict: dict = {key: [value] for key, value in experiment_dict.items()}

                experiment_dict.update({
                    "created_time_stamp": [datetime.now()],
                    "experiment_file_path": [os.path.basename(TrainingPipeline.experiment_file_path)]
                })

                experiment_report = pd.DataFrame(experiment_dict)

                os.makedirs(os.path.dirname(TrainingPipeline.experiment_file_path), exist_ok=True)
                if os.path.exists(TrainingPipeline.experiment_file_path):
                    experiment_report.to_csv(TrainingPipeline.experiment_file_path, index=False, header=False, mode="a")
                else:
                    experiment_report.to_csv(TrainingPipeline.experiment_file_path, index=False, header=True, mode="w")
            else:
                print("First start the experiment.")
        except Exception as e:
            raise CustomException(e,sys) from e

    @classmethod
    def get_experiments_status(cls, limit: int = 5) -> pd.DataFrame:
        try:
            if os.path.exists(TrainingPipeline.experiment_file_path):
                df = pd.read_csv(TrainingPipeline.experiment_file_path)
                limit = -1 * int(limit)
                return df[limit:].drop(columns=["experiment_file_path", "initialization_timestamp"], axis=1)
            else:
                return pd.DataFrame()
        except Exception as e:
            raise CustomException(e,sys) from e