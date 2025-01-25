from pydantic import BaseModel


class ModelNameConfig(BaseModel):
    """model coniguration"""
    model_name: str
    