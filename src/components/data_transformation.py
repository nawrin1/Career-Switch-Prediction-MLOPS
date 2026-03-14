import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, FunctionTransformer
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)



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

    def get_data_transformer_object(self) -> Pipeline:

        logging.info("get_data_transformer_object")
        try:  


            
            # High Missing -> Constant Imputer ("Unknown")
            high_missing_cols = ["gender", "company_type", "major_discipline"]
            
            # Low Missing -> Mode Imputer
            low_missing_cols_nominal = ["enrolled_university"]
            low_missing_col_ordinal = ["education_level"]      

            # Numerical
            num_cols = ['city_development_index', 'experience', 'last_new_job', 'training_hours', 'company_size_max']
            
            

            # pipelone-1
            high_miss_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("onehot", OneHotEncoder(sparse_output=False, drop="first", handle_unknown='ignore'))
            ])

            # pioline-2
            low_miss_nominal_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(sparse_output=False, drop="first", handle_unknown='ignore'))
            ])

            # pipeline-3
            edu_categories = ["Primary School", "High School", "Graduate", "Masters", "Phd", "Unknown"]
            ord_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(categories=[edu_categories], 
                                           handle_unknown='use_encoded_value', unknown_value=-1))
            ])

            # pipeline-4
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("power", PowerTransformer(method='yeo-johnson'))
            ])



            # column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("high_miss", high_miss_pipeline, high_missing_cols),
                    ("low_miss_nom", low_miss_nominal_pipeline, low_missing_cols_nominal),
                    ("ord_edu", ord_pipeline, low_missing_col_ordinal),
                    ("num", num_pipeline, num_cols)
                    
                ],
                remainder='passthrough'
            )

            
            final_pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("scaler", MinMaxScaler())
            ])

            logging.info("pipeline creation done")
            return final_pipeline

        except Exception as e:
            raise MyException(e, sys)


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            input_feature_train_df = train_df.drop([TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop([TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train and test df.")

            # Apply custom transformations in specified sequence
            input_feature_train_df= self._fix_city_column(input_feature_train_df)
            input_feature_train_df = self._map_relevant_experience_column(input_feature_train_df)
            # input_feature_train_df = self._fill_Missing_enrolled_university(input_feature_train_df)
            input_feature_train_df = self._fix_experience_column(input_feature_train_df)
            input_feature_train_df = self._fix_company_size_column(input_feature_train_df)
            input_feature_train_df = self._spliting_company_size_column(input_feature_train_df)
            input_feature_train_df = self._map_last_new_job_column(input_feature_train_df)
            input_feature_train_df = self._outlier_removal(input_feature_train_df)
            

            input_feature_test_df= self._fix_city_column(input_feature_test_df)
            input_feature_test_df = self._map_relevant_experience_column(input_feature_test_df)
            # input_feature_test_df = self._fill_Missing_enrolled_university(input_feature_test_df)
            input_feature_test_df = self._fix_experience_column(input_feature_test_df)
            input_feature_test_df = self._fix_company_size_column(input_feature_test_df)
            input_feature_test_df = self._spliting_company_size_column(input_feature_test_df)
            input_feature_test_df = self._map_last_new_job_column(input_feature_test_df)
            input_feature_test_df = self._outlier_removal(input_feature_test_df)
            
            logging.info("Custom transformations applied to train and test data")

            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            logging.info("Initializing transformation for Training-data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Initializing transformation for Testing-data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Transformation done end to end to train-test df.")

            logging.info("Applying SMOTEENN for handling imbalanced dataset.")

           

            

            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )
            logging.info("SMOTEENN applied to train-test df.")

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info("feature-target concatenation done for train-test df.")

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving transformation object and transformed files.")

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e



