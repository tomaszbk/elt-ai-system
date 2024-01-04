import pendulum

from airflow.decorators import dag

from datetime import datetime, timedelta
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.dummy import DummyOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator


@dag(
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    tags=["example"],
)
def laptop_el_dag():

    start = DummyOperator(task_id="start")

    ingest = DockerOperator(image="laptop_ingest",
                       task_id="laptop_ingest_task",
                       container_name="laptop_ingest_task",
                       auto_remove=False,
                       api_version='auto',
                        docker_url='unix://var/run/docker.sock',
                        network_mode='elt-ai-system_default',
                        
                       )



    transform = DockerOperator(image="laptop_dbt_transform",
                          task_id="laptop_dbt_transform_task",
                        container_name="laptop_dbt_transform_task",
                        auto_remove=False,
                        api_version='auto'
                       )
    

    finish = DummyOperator(task_id="finish")



    start >> ingest >> transform >> finish


laptop_el_dag()