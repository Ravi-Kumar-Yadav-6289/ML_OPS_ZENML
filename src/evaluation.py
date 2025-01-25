import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class EvalutaionStrategy(ABC):
    
    @abstractmethod
    def calc_score(self, y_true, y_test)-> None:
        # abstract method hai isko extend karna hai thik hai....
        pass


class RMSE(EvalutaionStrategy):
    
    def calc_score(self, y_true:np.ndarray, y_pred:np.ndarray)-> float:
        """
        Args : y_true, y_pred
        return : None
        """
        try:
            logging.info("Evaluating the model using RMSE")
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
            #logging.info(f"RMSE : {rmse}")
            return  rmse
        except Exception as e:
            logging.error(f"Error in evaluating the model {e}")
            raise e
        
class R2_score(EvalutaionStrategy):
    
    def calc_score(self, y_true, y_pred)-> float:
        """
        Args : y_true, y_pred
        return : None
        """
        try:
            logging.info("Evaluating the model using r2_score")
            r2 = r2_score(y_true, y_pred)
            #logging.info(f"RMSE : {rmse}")
            return  r2
        except Exception as e:
            logging.error(f"Error in evaluating the model {e}")
            raise e

class MSE(EvalutaionStrategy):
    
    def calc_score(self, y_true, y_pred)-> float:
        """
        Args : y_true, y_pred
        return : None
        """
        try:
            logging.info("Evaluating the model using mse")
            mse = mean_squared_error(y_true, y_pred)
            #logging.info(f"RMSE : {rmse}")
            return  mse
        except Exception as e:
            logging.error(f"Error in evaluating the model {e}")
            raise e