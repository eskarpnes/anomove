from etl import ETL
import pandas as pd
import numpy as np
from random import gauss

DATA_PATH = "/home/erlend/datasets"

etl = ETL(DATA_PATH, [128, 256, 512, 1024])
etl.cache = False
etl.load("CIMA")

infant = etl.cima["001"]

noisy_cima = {}

noisy_cima["001"] = infant

for i in range(100):
    data = infant["data"].copy()
    for key, val in data.items():
        if key == "frame":
            continue
        noise = pd.Series([gauss(0.0, 0.005) for i in range(len(val))])
        data[key] = val.add(noise)
    noisy_cima[f"noise_{i}"] = {"data": data, "label": 0, "fps": 24}


etl.cima = noisy_cima

etl.preprocess_pooled(20)

true_infant = etl.cima["001"]

raw_differences = pd.DataFrame()
z_differences = pd.DataFrame()

for infant_key, infant_val in etl.cima.items():
    if infant_key == "001":
        continue
    data = infant_val["data"]
    z_data = infant_val["z_interpolation"]
    raw_data_noise = data.sub(true_infant["data"])
    z_data_noise = z_data.sub(true_infant["z_interpolation"])

    raw_differences = raw_differences.append(raw_data_noise)
    z_differences = z_differences.append(z_data_noise)

    # for key, val in data.items():
    #     noise = val.sub(true_infant["data"][key])
    #     print(f"Infant {infant_key} and point {key} got mean {noise.mean()} and std {noise.std()}")
    #
    # data = infant_val["z_interpolation"]
    # for key, val in data.items():
    #     noise = val.sub(true_infant["z_interpolation"][key])
    #     print(f"Infant {infant_key} and point {key} got mean {noise.mean()} and std {noise.std()}")

pd.options.display.max_columns = 50
print(raw_differences.describe())
print(z_differences.describe())

raw_differences = raw_differences.reset_index()
z_differences = z_differences.reset_index()

raw_differences.to_json("raw_differences.json")
z_differences.to_json("z_differences.json")
