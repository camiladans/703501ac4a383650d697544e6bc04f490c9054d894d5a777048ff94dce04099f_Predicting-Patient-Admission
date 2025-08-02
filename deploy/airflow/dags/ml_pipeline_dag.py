from airflow.decorators import dag, task
from datetime import datetime
import pandas as pd
from joblib import load
import sys

sys.path.append("/app/src")

from data_preprocessing import preprocess_data
from feature_engineering import engineer_features
from model_training import train_model, save_model
from evaluation import evaluate_model

@dag(
    dag_id='ml_pipeline_dag',
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=['ml'],
)
def ml_pipeline():

    @task()
    def step_preprocess():
        train, test = preprocess_data("/app/data/raw/data-ori.csv")
        train_path = "/app/data/train.pkl"
        test_path = "/app/data/test.pkl"
        train.to_pickle(train_path)
        test.to_pickle(test_path)
        return {"train_path": train_path, "test_path": test_path}

    @task()
    def step_engineer(paths: dict):
        train = pd.read_pickle(paths["train_path"])
        test = pd.read_pickle(paths["test_path"])

        train = engineer_features(train)
        test = engineer_features(test)

        train_feat_path = "/app/data/train_feat.pkl"
        test_feat_path = "/app/data/test_feat.pkl"
        train.to_pickle(train_feat_path)
        test.to_pickle(test_feat_path)
        return {"train_feat_path": train_feat_path, "test_feat_path": test_feat_path}

    @task()
    def step_train(paths: dict):
        train = pd.read_pickle(paths["train_feat_path"])
        model, _ = train_model(train, model_type="logreg")
        model_path = "/app/models/model.pkl"
        save_model(model, model_path)
        return model_path

    @task()
    def step_evaluate(paths: dict, model_path: str):
        test = pd.read_pickle(paths["test_feat_path"])
        model = load(model_path)
        evaluate_model(model, test,
                       report_path="/app/reports/metrics.txt")

    # Task dependencies
    raw_paths = step_preprocess()
    feat_paths = step_engineer(raw_paths)
    model_file = step_train(feat_paths)
    step_evaluate(feat_paths, model_file)

# Instantiate the DAG
ml_pipeline()
