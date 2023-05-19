import os, sys
import yaml 
import numpy as np
import dill
import pandas as pd
from visa.constant import *
from visa.exception import CustomException
from visa.logger import logging


def read_yaml_file(file_path:str) -> dict:
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e, sys) from e