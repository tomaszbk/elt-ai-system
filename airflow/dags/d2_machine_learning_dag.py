import pendulum

from airflow.decorators import dag

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.dummy import DummyOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from docker.types import Mount 


@dag(
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    tags=["example"],
)
def laptop_ml_train_dag():

    start = DummyOperator(task_id="start")



    train_model = DockerOperator(image="laptop_ml_train",
                       task_id="laptop_ingest_task",
                       container_name=f"laptop_ml_train_task{pendulum.now().format('YMMDDHHmm')}",
                       auto_remove=False,
                       api_version='auto',
                        docker_url='unix://var/run/docker.sock',
                        network_mode='elt-ai-system_default',
                        mounts = [
                                Mount(
                                source = r"C:\Users\tzbk\CODIGOS\data_engineer\elt-ai-system\ml_model",
                                target = "/ml_model",
                                type='bind'
                                    )
                                   ]
                       )
    

    finish = DummyOperator(task_id="finish")



    start >>  train_model >>  finish


laptop_ml_train_dag()