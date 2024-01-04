import json

import pendulum

from airflow.decorators import dag, task

from datetime import datetime, timedelta
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators import BashOperator
from airflow.operators import DummyOperator


@dag(
    schedule=None,
    start_date=None,
    catchup=False,
    tags=["example"],
)
def laptop_el_dag():

    @task()
    def ingest():
        data_string = '{"1001": 301.27, "1002": 433.21, "1003": 502.22}'
        BashOperator
        order_data_dict = json.loads(data_string)
        DockerOperator
        return order_data_dict


    @task()
    def transform(total_order_value: float):
        print(f"Total order value is: {total_order_value:.2f}")


    ingest_laptop_data = ingest()
    transform_laptop_data = transform(ingest_laptop_data)

laptop_el_dag()