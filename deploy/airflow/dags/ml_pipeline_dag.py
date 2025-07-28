from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys

sys.path.append("/app/src")

from data_preprocessing import preprocess_data
from feature_engineering import engineer_features
from model_training import train_model, save_model
from evaluation import evaluate_model

default_args = {
    'start_date': datetime(2025, 1, 1),
    'catchup': False,
}

with DAG(
    dag_id='ml_pipeline_dag',
    default_args=default_args,
    schedule_interval=None,
    tags=['ml'],
) as dag:

    def step_preprocess(**context):
        train, test = preprocess_data("/app/data/raw/data-ori.xlsx")
        train_path = "/app/data/train.pkl"
        test_path = "/app/data/test.pkl"
        train.to_pickle(train_path)
        test.to_pickle(test_path)

        context['ti'].xcom_push(key="train_path", value=train_path)
        context['ti'].xcom_push(key="test_path", value=test_path)

    def step_engineer(**context):
        import pandas as pd
        train_path = context['ti'].xcom_pull(key="train_path", task_ids="preprocess")
        test_path = context['ti'].xcom_pull(key="test_path", task_ids="preprocess")

        train = pd.read_pickle(train_path)
        test = pd.read_pickle(test_path)
        train = engineer_features(train)
        test = engineer_features(test)

        train_feat_path = "/app/data/train_feat.pkl"
        test_feat_path = "/app/data/test_feat.pkl"
        train.to_pickle(train_feat_path)
        test.to_pickle(test_feat_path)

        context['ti'].xcom_push(key="train_feat_path", value=train_feat_path)
        context['ti'].xcom_push(key="test_feat_path", value=test_feat_path)

    def step_train(**context):
        import pandas as pd
        train_feat_path = context['ti'].xcom_pull(key="train_feat_path", task_ids="engineer")
        train = pd.read_pickle(train_feat_path)

        model, _ = train_model(train, model_type="logreg")
        model_path = "/app/models/model.pkl"
        save_model(model, model_path)

        context['ti'].xcom_push(key="model_path", value=model_path)

    def step_evaluate(**context):
        import pandas as pd
        from joblib import load
        test_feat_path = context['ti'].xcom_pull(key="test_feat_path", task_ids="engineer")
        model_path = context['ti'].xcom_pull(key="model_path", task_ids="train")

        test = pd.read_pickle(test_feat_path)
        model = load(model_path)

        evaluate_model(model, test,
                       report_path="/app/reports/metrics.txt",
                       shap_path="/app/reports/shap_summary.html")

    preprocess_task = PythonOperator(task_id="preprocess", python_callable=step_preprocess)
    engineer_task = PythonOperator(task_id="engineer", python_callable=step_engineer)
    train_task = PythonOperator(task_id="train", python_callable=step_train)
    evaluate_task = PythonOperator(task_id="evaluate", python_callable=step_evaluate)

    preprocess_task >> engineer_task >> train_task >> evaluate_task
