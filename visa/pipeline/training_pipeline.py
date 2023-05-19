import os, sys
import pandas as pd
import numpy as np
from visa.constant import *
from visa.logger import logging
from visa.entity.config_entity import DataIngestionConfig
from visa.entity.artifact_entity import DataIngestionArtifact
from visa.exception import CustomException
from datetime import date
from collections import namedtuple
from visa.config.configuration import Configuration
from visa.components.data_ingestion import DataIngestion

class Pipeline():
    def __init__(self, config: Configuration = Configuration())->None:
        try:
            self.config = config
        except Exception as e:
            raise CustomException(e,sys) from e  
        
    def start_data_ingestion(self)-> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config= self.config.get_data_ingestion_config())
            return data_ingestion.initate_data_ingestion()
        except Exception as e:
            raise CustomException(e,sys) from e  
        
    
        
    def run_pipeline(self):
        try:
            # Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()
        except Exception as e:
            raise CustomException(e,sys) from e  