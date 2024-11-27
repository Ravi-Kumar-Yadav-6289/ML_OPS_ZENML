import pandas as pd
import numpy as np
import logging
from zenml import step

@step
def train(df: pd.DataFrame) ->None:
    pass

