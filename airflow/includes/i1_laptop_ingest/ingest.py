import subprocess
import pandas as pd
import os

def extract():
    print("extracting data...")
    r = subprocess.run("extract.sh", shell=True, capture_output=True, executable="/bin/bash")
    print(r.stdout)


def load():
    laptops_df = pd.read_csv("laptops.csv")
    laptops_df.to_sql("laptop_raw", "postgres:///laptops.db")


if __name__ == "__main__":
    extract()
    load()
