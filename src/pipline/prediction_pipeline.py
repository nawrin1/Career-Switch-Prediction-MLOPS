import sys
import pandas as pd
import numpy as np
from pandas import DataFrame
from src.entity.config_entity import CareerPredictorConfig 
from src.entity.s3_estimator import CareerSwitchEstimator 
from src.exception import MyException
from src.logger import logging

class CareerData:
    def __init__(self,
                city,
                city_development_index,
                gender,
                relevent_experience,
                enrolled_university,
                education_level,
                major_discipline,
                experience,
                company_type,
                last_new_job,
                training_hours,
                company_size_max
                ):
        """
        Career Data constructor
        
        """
        try:
            self.city = city
            self.city_development_index = city_development_index
            self.gender = gender
            self.relevent_experience = relevent_experience
            self.enrolled_university = enrolled_university
            self.education_level = education_level
            self.major_discipline = major_discipline
            self.experience = experience
            self.company_type = company_type
            self.last_new_job = last_new_job
            self.training_hours = training_hours
            self.company_size_max = company_size_max

        except Exception as e:
            raise MyException(e, sys) from e

    def get_career_input_data_frame(self) -> DataFrame:
        
        try:
            career_input_dict = self.get_career_data_as_dict()
            return DataFrame(career_input_dict)
        except Exception as e:
            raise MyException(e, sys) from e

    def get_career_data_as_dict(self):
        
        logging.info("Creating dictionary for career data")

        try:
            input_data = {
                "city": [self.city],
                "city_development_index": [self.city_development_index],
                "gender": [self.gender],
                "relevent_experience": [self.relevent_experience],
                "enrolled_university": [self.enrolled_university],
                "education_level": [self.education_level],
                "major_discipline": [self.major_discipline],
                "experience": [self.experience],
                "company_type": [self.company_type],
                "last_new_job": [self.last_new_job],
                "training_hours": [self.training_hours],
                "company_size_max": [self.company_size_max],
                "enrollee_id": [0],         #any random value will work here since we are not using this column for prediction
                "company_size_min": [0]# any random value will work here since we are not using this column for prediction
            }

            logging.info("Career data dictionary created successfully and ready for model prediction")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e

class CareerClassifier:
    def __init__(self, prediction_pipeline_config: CareerPredictorConfig = CareerPredictorConfig()) -> None:
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)

    def _apply_manual_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This method is responsible for applying manual cleaning on the input dataframe before prediction.
        """
        try:
            logging.info("প্রstarting manual cleaning of input data for prediction")
            df = df.copy()

            #  Experience Fix
            def fix_experience(val):
                if val == "<1": return "0"
                if val == ">20": return "21"
                return val
            df["experience"] = df["experience"].apply(fix_experience).astype(float)

           
            df["company_size_max"] = df["company_size_max"].str.split("-", expand=True)[1]
            df["company_size_max"] = pd.to_numeric(df["company_size_max"], errors="coerce")

            #  Last New Job Fix
            def fix_last_job(val):
                if val == ">4": return "5"
                if val == "never": return 0
                return val
            df["last_new_job"] = df["last_new_job"].apply(fix_last_job).astype(float)

            #  City Fix (city_103 -> 103)
            df["city"] = df["city"].str.split("_", expand=True)[1].astype(int)

            # Relevant Experience Fix (Mapping)
            df['relevent_experience'] = df['relevent_experience'].map(
                {'No relevent experience': 0, 'Has relevent experience': 1}).fillna(0).astype(int)
            
            
            df["enrollee_id"] = 0
            df["company_size_min"] = 0

            return df
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe: pd.DataFrame):
        try:
            logging.info("production started for prediction")
            
            # S3 Estimator 
            model = CareerSwitchEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            
           
            cleaned_df = self._apply_manual_cleaning(dataframe)
            
            
            result = model.predict(cleaned_df)
            return result
        
        except Exception as e:
            raise MyException(e, sys)