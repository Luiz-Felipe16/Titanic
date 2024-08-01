from airflow import DAG
from airflow.operators.python_operator import PythonOperator # type: ignore
from datetime import datetime
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

def ingest_data():
    os.system('python data_ingestion.py')

def run_etl():
    os.system('python etl.py')

def generate_ml_dataset():
    os.system('python generate_ml_dataset.py')

def train_model():
    os.system('python train_model.py')

with DAG('main_dag', default_args=default_args, schedule_interval='@daily') as dag:
    ingestion_task = PythonOperator(
        task_id='ingest_data',
        python_callable=ingest_data,
    )

    etl_task = PythonOperator(
        task_id='run_etl',
        python_callable=run_etl,
    )
    
    ml_dataset_task = PythonOperator(
        task_id='generate_ml_dataset',
        python_callable=generate_ml_dataset,
    )

    training_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    ingestion_task >> etl_task >> ml_dataset_task >> training_task
