import pandas as pd
import numpy as np
import logging
from zenml import step
from src.model_dev import LinearModel
from sklearn.base import RegressorMixin
from steps.config import ModelNameConfig


@step
def train(x_train,y_train,model_config) -> any:
    model = None
    try:
        if model_config.model_name == "LinearRegression":
            model = LinearModel()
            trained_model = model.train(x_train,y_train)
            logging.info("Model trained successfully")
            return trained_model
        # here i can add mode if and give different models.
        else:
            raise ValueError("Model not implemented")
    except Exception as e:
        logging.error(f"Error in training the model {e}")
        raise e
    