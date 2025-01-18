import pandas as pd
import numpy as np
import logging
from zenml import step
from sklearn.base import RegressorMixin
from src.evaluation import MSE, R2_score, RMSE
@step
def evaluate(model:RegressorMixin, x_test, y_test) ->None:
    '''
    Args : model, x_test, y_test
    return : None
    '''
    try:
        y_pred = model.predict(x_test)
        logging.info(f"Model evaluation completed")
        mse = MSE().evaluate(y_test, y_pred)
        rmse = RMSE().evaluate(y_test, y_pred)
        r2=R2_score().evaluate(y_test, y_pred)
        return mse
    
    except Exception as e:
        logging.error(f"Error in evaluating the model {e}")
        raise e

