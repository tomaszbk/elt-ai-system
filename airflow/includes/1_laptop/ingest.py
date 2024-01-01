import subprocess
import pandas as pd

def extract():
    print("extracting data...")
    r = subprocess.run("extract.sh", shell=True, capture_output=True, executable="/bin/bash")
    print(r.stdout)
def load():
    pd.read_csv("laptops.csv")
    pd.to_sql("laptops", "sqlite:///laptops.db")


if __name__ == "__main__":
    extract()
    load()

print(__name__)