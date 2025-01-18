import logging
import numpy as np
import pandas as pd
from typing import Union
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

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
                ]
            )

            data['product_weight_g'].fillna(data['product_weight_g'].median(), inplace=True)
            data['product_length_cm'].fillna(data['product_length_cm'].median(), inplace=True)
            data['product_height_cm'].fillna(data['product_height_cm'].median(), inplace=True)
            data['product_width_cm'].fillna(data['product_width_cm'].median(), inplace=True)
            data['review_comment_message'].fillna("No Review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            drop_cols = ['customer_zip_code_prefix','order_item_id']
            data = data.drop(columns=drop_cols)
            return data
        except Exception as e:
            logging.error(f"Error in preprocessing the data {e}")
            raise e

class DataSplitStrategy(DataStrategy):
    def handle_data(self, data:pd.DataFrame)-> Union[pd.DataFrame, pd.Series]:
        """
        To do the train test split.
        """
        try:
            x =  data.drop(columns=['review_score'])
            y = data['review_score']
            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
            return x_train,x_test,y_train,y_test
        except Exception as e:
            logging.error(f"Error in train test split {e}")
            raise e

class DataCleaning:
    """
    This will utilize the classes above to process the data and make the train test split.
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
    
    def handle_data(self):
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"error in handling the data {e}")
            raise e
    
if __name__ == "__main__":
    pass
    # data = pd.read_csv(r"D:\ML_OPS_ZENML\data\olist_customers_dataset.csv")
    # data_cleaning = DataCleaning(data, DataPreprocessingStrat())
    # data_cleaning.handle_data()
    # the above lines can help me call any pre processing strategy defined above.
