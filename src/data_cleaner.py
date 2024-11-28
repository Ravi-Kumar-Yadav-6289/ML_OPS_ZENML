import logging
import numpy as np
import pandas as pd
from typing import Union
from abc import ABC, abstractmethod


class DataStrategy(ABC):

    @abstractmethod
    def handle_data(self, data:pd.DataFrame)-> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessingStrat(DataStrategy):
    
    def handle_data(self, data:pd.DataFrame)-> Union[pd.DataFrame, pd.Series]:
        # pre process data
        try:
            data = data.drop(
                columns=[
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_testimony"
                ]
            )

            data['product_weight_g'].fillna(data['product_weight_g'].median(), inplace=True)
            data['product_length_cm'].fillna(data['product_length_cm'].median(), inplace=True)
            data['product_height_cm'].fillna(data['product_height_cm'].median(), inplace=True)
            data['product_width_cm'].fillna(data['product_width_cm'].median(), inplace=True)
            data['review_comment_message'].fillna("No Review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            drop_cols = ['cusotmer_zip_code_prefix','order_item_id']
            data = data.drop(columns=drop_cols)
            return data
        except Exception as e:
            logging.error(f"{"Error in preprocessing the data {e}"}")
            raise e

