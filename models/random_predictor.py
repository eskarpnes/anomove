import random
import pandas as pd


def predict(infant):
    for column_name in list(infant):
        predictions = pd.Series([[random.randint(-1, 1)] * random.randint(30, 128) for i in range(len(infant))])
        predictions = [i for sublist in predictions for i in sublist]
        infant[column_name + "_pred"] = predictions[:len(infant)]
    return infant.iloc[:, 8:]
