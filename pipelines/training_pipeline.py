from zenml import pipeline
from steps.data_ingestion import ingest_data
from steps.preprocess import pre_process
from steps.mode_train import train
from steps.eval import evaluate

@pipeline
def training_pipline(data_path:str):
    df = ingest_data(data_path)
    pre_process(df)
    train(df)
    evaluate(df)
    