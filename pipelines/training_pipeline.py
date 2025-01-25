from zenml import pipeline
from steps.data_ingestion import ingest_data
from steps.preprocess import pre_process
from steps.model_train import train
from steps.eval import evaluate
from steps.config import ModelNameConfig

@pipeline
def training_pipline(data_path:str):
    df = ingest_data(data_path)
    x_train,x_test,y_train,y_test = pre_process(df)
    #print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
    config = ModelNameConfig(model_name="LinearRegression")
    model=train(x_train,y_train,config)
    mse,rmse,r2=evaluate(model,x_test,y_test)