import pandas as pd
import numpy as np
import logging
from zenml import step


class IngestData:

    def __init__(self,path):
        self.data_path = path

    def load_data(self):
        logging.info(f"loading data from {self.data_path}")
        return pd.read_csv(self.data_path)
    

@step
def ingest_data(data_path: str)->pd.DataFrame:
    '''
    Args : data path

    returns : data frame
    '''

    try:
        ingest = IngestData(data_path)
        df = ingest.load_data()
        return df
    except Exception as e:
        logging.error(f"Error while loading the data : {e}")
        raise e
