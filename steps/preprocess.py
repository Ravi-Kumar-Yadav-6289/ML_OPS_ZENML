import pandas as pd
import numpy as np
import logging
from zenml import step
from src.data_cleaner import DataCleaning, DataPreprocessingStrat,DataSplitStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def pre_process(df: pd.DataFrame) ->Tuple[
    Annotated[pd.DataFrame,"x_train"],
    Annotated[pd.DataFrame,"x_test"],
    Annotated[pd.DataFrame,"y_train"],
    Annotated[pd.DataFrame,"y_test"],
]:
    """
    Args : data frame.
    return : data split as train test.
    """
    try:
        pre_process_strat = DataPreprocessingStrat()
        pre_prcessing = DataCleaning(df,pre_process_strat)
        clean_data = pre_prcessing.handle_data()

        divide_strat =  DataSplitStrategy()
        splitting = DataCleaning(clean_data, DataSplitStrategy())
        x_train, x_test,y_train,y_test =  splitting.handle()
        #return x_train, x_test,y_train,y_test
        logging.info("Data cleaning and splitting complete")
    except Exception as e:
        logging.error(f"error occoured while performing pipeline step tp clean{e}")
        raise e


