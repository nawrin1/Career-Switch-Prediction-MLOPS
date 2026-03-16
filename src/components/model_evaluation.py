from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object
import sys
import pandas as pd
from typing import Optional
from src.entity.s3_estimator import CareerSwitchEstimator
from dataclasses import dataclass
import numpy as np

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) :
        """
        Method Name :   get_best_model
        Description :   This function is used to get model from production stage.
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            careerSwitch_estimator = CareerSwitchEstimator(bucket_name=bucket_name,
                                               model_path=model_path)

            if careerSwitch_estimator.is_model_present(model_path=model_path):
                return careerSwitch_estimator
            return None
        except Exception as e:
            raise  MyException(e,sys)
        
    def _fix_city_column(self, df):
        """Fix the 'City' column by mapping 'City_X' to "X" and 'City_Y' to "Y"."""
        logging.info("fixing city column")
        df["city"] = df["city"].str.split("_", expand=True)[1]
        df["city"] = df["city"].astype("int")
        return df
    def _map_relevant_experience_column(self, df):

        """Map 'Relevant_Experience' column to binary values."""
        logging.info("Mapping 'Relevant_Experience' column to binary values")
        # df["relevent_experience"] = np.where(df["relevent_experience"] == 'Has relevent experience', 1, 0)
        df['relevent_experience'] = df['relevent_experience'].map({'No relevent experience': 0, 'Has relevent experience': 1}).astype(int)
        return df
    # def _fill_Missing_enrolled_university(self, df):
    #     """Fill missing values in 'Enrolled_University' column with the mode."""
    #     logging.info("Filling missing values in 'Enrolled_University' column with mode")
    #     # imp_mode  = SimpleImputer(strategy='most_frequent')
    #     # df["enrolled_university"] = imp_mode.fit_transform(df[["enrolled_university"]]).ravel()
    #     df["enrolled_university"].fillna(df["enrolled_university"].mode()[0], inplace=True)


        
        return df
    def _fix_experience_column(self, df):
        """Fix the 'Experience' column by replacing '>20' with 21 and '<1' with 0."""
        logging.info("Fixing 'Experience' column by replacing '>20' with 21 and '<1' with 0")
        def Experience(val):
            if(val  ==  "<1"):
                return "0"
            if(val  ==  ">20"):
                return "21"
            else:
                return val
        df["experience"] = df["experience"].apply(Experience)
        df["experience"]=df["experience"].astype("float")
        # df['experience'] = df['experience'].replace({'>20': 20, '<1': 0}).astype(int)
        return df
    def _fix_company_size_column(self, df):
        """Fix the 'Company_Size' column by replacing '10000+' with 10000-20000 and 'Oct-49' with 10-49."""
        logging.info("Fixing 'Company_Size' column by replacing '10000+' with 10000-20000 and 'Oct-49' with 10-49 and '<10' with 1-9")
        def CompanySize(val):
            if(val == "10000+"):
                return "10000-20000"
            elif(val== "Oct-49"):
                return "10-49"
            elif(val == "<10"):
                return "1-9"
            else:
                return val
        df["company_size"] = df["company_size"].apply(CompanySize)
        return df
    
    def _spliting_company_size_column(self, df):
        """Split the 'Company_Size' column into two separate columns: 'Company_Size_Min' and 'Company_Size_Max'."""
        logging.info("Splitting 'Company_Size' column into 'Company_Size_Min' and 'Company_Size_Max'")
        df["company_size_min"] = df["company_size"].str.split("-", expand=True)[0]
        df["company_size_max"] = df["company_size"].str.split("-", expand=True)[1]
        df["company_size_min"] = pd.to_numeric(df["company_size_min"], errors="coerce")
        df["company_size_max"] = pd.to_numeric(df["company_size_max"], errors="coerce")
        
        df[["company_size_min", "company_size_max"]] = df[["company_size_min", "company_size_max"]].fillna(0).astype(int)


        df.drop("company_size", axis=1, inplace=True)
        return df
    def _map_last_new_job_column(self, df):
        """Map 'last_new_job' column to numeric values, replacing 'never' with 0 and '>4' with 4."""
        logging.info("Mapping 'last_new_job' column to numeric values, replacing 'never' with 0 and '>4' with 4")
        def last_new_job(val):
            if(val == ">4"):
                return "5"
            if val == "never": return 0
            else:
                return val
        df["last_new_job"] = df["last_new_job"].apply(last_new_job)
        df["last_new_job"] = pd.to_numeric(df["last_new_job"], errors="coerce")
        return df
   
    def _outlier_removal(self, df):
        percentile25 = df['training_hours'].quantile(0.25)
        percentile75 = df['training_hours'].quantile(0.75)
        iqr = percentile75 - percentile25

        upper_limit = percentile75 + 1.5 * iqr
        lower_limit = percentile25 - 1.5 * iqr


        df["training_hours"] = np.where(
            df["training_hours"] > upper_limit, upper_limit,
            np.where(df["training_hours"] < lower_limit, lower_limit, df["training_hours"])
        )
        return df

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            logging.info("Test data loaded and now transforming it for prediction...")

            x = self._fix_city_column(x)
            x = self._map_relevant_experience_column(x)
            # x = self._fill_Missing_enrolled_university(x)
            x = self._fix_experience_column(x)
            x = self._fix_company_size_column(x)
            x = self._spliting_company_size_column(x)
            x = self._map_last_new_job_column(x)
            x = self._outlier_removal(x)
            logging.info("Test data transformed successfully.")

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded/exists.")
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"F1_Score for this model: {trained_model_f1_score}")

            best_model_f1_score=None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info(f"Computing F1_Score for production model..")
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logging.info(f"F1_Score-Production Model: {best_model_f1_score}, F1_Score-New Trained Model: {trained_model_f1_score}")
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference=trained_model_f1_score - tmp_best_model_score
                                           )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e