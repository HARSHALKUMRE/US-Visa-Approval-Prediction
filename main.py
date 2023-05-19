import os, sys
import pandas as pd
import numpy as np
from visa.constant import *
from visa.logger import logging
from visa.exception import CustomException
from visa.pipeline.training_pipeline import Pipeline

def main():
    try:
        pipeline = Pipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.error(f"{e}")

if __name__ == "__main__":
    main() 