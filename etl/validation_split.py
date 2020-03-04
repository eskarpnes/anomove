from etl import ETL
import os

DATASET_PATH = "/home/login/datasets"
DATASET_NAME = "CIMA"
DATASET_SIZE = 377
validation_size = 0.2

validation_size = int(DATASET_SIZE * validation_size)

validation_etl = ETL("/home/login/datasets", [], size=validation_size)
validation_etl.load(DATASET_NAME)

validation_path = os.path.join(DATASET_PATH, DATASET_NAME, "validation")
if not os.path.exists(validation_path):
    os.mkdir(validation_path)
data_path = os.path.join(DATASET_PATH, DATASET_NAME, "data")

validation_set = validation_etl.cima

for key, item in validation_set.items():
    print(f"Moving {key} to validation.")
    old_path = os.path.join(data_path, key + ".csv")
    new_path = os.path.join(validation_path, key + ".csv")
    os.rename(old_path, new_path)