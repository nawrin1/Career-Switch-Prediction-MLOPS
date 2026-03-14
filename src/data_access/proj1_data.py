import sys
import pandas as pd
import numpy as np
from typing import Optional

from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME
from src.exception import MyException
class CareerSwitchData:
    """
    class to pull data from mongodb
    """

    def __init__(self):
        try:
            self.mongo_client=MongoDBClient()

        except Exception as e:
            raise MyException(e, sys)
        
    def export_collection_as_dataframe(self,collection_name="CareerMLOPS-Data",database_name="CareerSwitchPredictionMLOPS"):
        """
        Method to export collection data as dataframe
        Returns: Datafram containing collection of data with "_id" column removed
        """
        try:
            if database_name != self.mongo_client.database_name:
                database = self.mongo_client.client[database_name]
            else:
                database = self.mongo_client.database

            collection = database[collection_name]


           
            print("Fetching data from mongoDB")
            df = pd.DataFrame(list(collection.find()))
            print(f"Data fecthed with len: {len(df)}")
            if "_id" in df.columns.to_list():
                df = df.drop(["_id"], axis=1)
            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise MyException(e, sys)