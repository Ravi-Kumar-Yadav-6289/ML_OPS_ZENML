import logging
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
from sklearn.linear_model import LinearRegression

class ModelImplementation(ABC):
    
    @abstractmethod
    def train(self, x_train:pd.DataFrame, y_train:pd.DataFrame)-> Union[LinearRegression, None]:
        pass



class LinearModel(ModelImplementation):
    
    def train(self, x_train:pd.DataFrame, y_train:pd.DataFrame, **kwargs)-> Union[LinearRegression, None]:
        """
        Args : x_train, y_train
        return : model
        """
        try:
            model = LinearRegression()
            model.fit(x_train, y_train)
            logging.info("Model trained successfully")
            return model
        except Exception as e:
            logging.error(f"Error in training the model {e}")
            raise e