import json

import pendulum

from airflow.decorators import dag, task

@dag(
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    tags=["example"],
)
def laptop_el_dag():

    @task()
    def extract():
        data_string = '{"1001": 301.27, "1002": 433.21, "1003": 502.22}'

        order_data_dict = json.loads(data_string)
        return order_data_dict


    @task()
    def load(total_order_value: float):
        print(f"Total order value is: {total_order_value:.2f}")


    extract_laptop_data = extract()
    load_laptop_data = load(extract_laptop_data)

laptop_el_dag()