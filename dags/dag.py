from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='dag',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule_interval='@monthly',
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
) as dag:

    data_preprocessing = BashOperator(
        task_id='data_preprocessing',
        bash_command=(
            'cd /opt/airflow && '
            'python src/preprocess.py --config configs/default_preprocess.json'
        ),
    )

    model_training = BashOperator(
        task_id='model_training',
        bash_command=(
            'cd /opt/airflow && '
            'python src/train_model.py --config configs/default_train.json'
        ),
    )

    data_preprocessing >> model_training