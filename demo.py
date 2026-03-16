from src.cloud_storage.aws_storage import SimpleStorageService
from src.logger import logging
from src.data_access.proj1_data import CareerSwitchData
from src.components.data_ingestion import DataIngestion
from src.configuration.mongo_db_connection import MongoDBClient
from src.entity.config_entity import DataIngestionConfig
# data_ingestion_config = DataIngestionConfig()

# data_ingestion = DataIngestion(data_ingestion_config)
# data_ingestion_artifact = data_ingestion.initiate_data_ingestion()


from src.pipline.training_pipeline import TrainPipeline

pipline = TrainPipeline()
pipline.run_pipeline()

# aws=SimpleStorageService()
# print(aws.get_bucket("career-switch-142333318099-ap-southeast-1-an"))
