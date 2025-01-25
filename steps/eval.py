import pandas as pd
import numpy as np
import logging
from zenml import step
from sklearn.base import RegressorMixin
from src.evaluation import MSE, R2_score, RMSE
from typing import Tuple
from typing_extensions import Annotated



@step
def evaluate(model:RegressorMixin, x_test, y_test) -> Tuple[
    Annotated[float, "MSE"],
    Annotated[float, "RMSE"],
    Annotated[float, "R2_score"]
]:
    '''
    Args : model, x_test, y_test
    return : None
    '''
    try:
        y_pred = model.predict(x_test)
        logging.info(f"Model evaluation completed")
        mse = MSE().calc_score(y_test, y_pred)
        rmse = RMSE().calc_score(y_test, y_pred)
        r2=R2_score().calc_score(y_test, y_pred)
        return mse,rmse,r2
    
    except Exception as e:
        logging.error(f"Error in evaluating the model {e}")
        raise e

