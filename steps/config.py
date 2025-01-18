from zenml import BaseParameters

class ModelNameConfig(BaseParameters):
    """model coniguration"""
    model_name: str = 'LinearRegression'
    