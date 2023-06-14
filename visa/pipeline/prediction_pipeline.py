import os
import sys

from visa.exception import CustomException
from visa.utils.main_utils import load_object

import pandas as pd
import numpy as np

class USVisaData:

    def __init__(self,
                 continent: str,
                 education_of_employee: str,
                 has_job_experience: str,
                 requires_job_training: str,
                 no_of_employees: int,
                 company_age: int,
                 region_of_employment: str,
                 prevailing_wage: float,
                 unit_of_wage: str, 
                 full_time_position: str,
                 case_status: str = None
                 ):
        try:
            self.continent = continent
            self.education_of_employee = education_of_employee
            self.has_job_experience = has_job_experience
            self.requires_job_training = requires_job_training
            self.no_of_employees = no_of_employees
            self.company_age = company_age
            self.region_of_employment = region_of_employment
            self.prevailing_wage = prevailing_wage
            self.unit_of_wage = unit_of_wage
            self.full_time_position = full_time_position
            self.case_status = case_status
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_us_visa_input_data_frame(self):
        try:
            us_visa_input_dict = self.get_us_visa_data_as_dict()
            return pd.DataFrame(us_visa_input_dict)
        except Exception as e:
            raise CustomException(e, sys) from e


    def get_us_visa_data_as_dict(self):
        try:
            input_data = {
                "continent":[self.continent],
                "education_of_employee":[self.education_of_employee],
                "has_job_experience":[self.has_job_experience],
                "requires_job_training":[self.requires_job_training],
                "no_of_employees":[self.no_of_employees],
                "company_age":[self.company_age],
                "region_of_employment":[self.region_of_employment],
                "prevailing_wage":[self.prevailing_wage],
                "unit_of_wage":[self.unit_of_wage],
                "full_time_position":[self.full_time_position]
            }
        except Exception as e:
            raise CustomException(e, sys) from e


class USVisaPredictor:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise CustomException(e, sys) from e

    def get_latest_model_path(self):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            return latest_model_path
        except Exception as e:
            raise CustomExcetion(e, sys) from e

    def predict(self, X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            case_status = model.predict(X)
            return case_status
        except Exception as e:
            raise CustomException(e, sys) from e

   