import pandas as pd
import numpy as np
import os
from time import gmtime, strftime
import joblib
import sys

sys.path.append("../")
from etl.etl import ETL


class Predictor:

    def __init__(self):
        self.model = None

    def load_model(self, model_name):
        path = os.path.join("models/saved_models", model_name + ".joblib")
        self.model = joblib.load(path)

    def save_model(self, model_name):
        if not os.path.exists("saved_models"):
            os.mkdir("saved_models")
        path = os.path.join("saved_models", model_name + ".joblib")
        joblib.dump(self.model, path)

    def predict(self, data_path, infant_id):
        print(f"Predicting infant {infant_id} - {strftime('%H:%M:%S', gmtime())}")
        window_sizes = [128, 256, 512, 1024]
        etl = ETL(data_path, window_sizes)
        etl.load_infant(infant_id)
        print(f"Preprocessing the data - {strftime('%H:%M:%S', gmtime())}")
        etl.preprocess_pooled()

        angles = ["right_shoulder", "left_shoulder", "right_elbow", "left_elbow", "right_hip", "left_hip", "right_knee",
                  "left_knee"]
        predictions = {}
        for angle in angles:
            predictions[angle] = pd.Series([[] for i in range(len(etl.cima[infant_id]["data"]))])

        print(f"Generating fourier data - {strftime('%H:%M:%S', gmtime())}")
        for window_size in window_sizes:
            for angle in angles:
                dataframe = etl.generate_fourier_data(window_size, angle, window_size)
                data_features = pd.DataFrame(dataframe.data.tolist())
                data_transformed = self.model[window_size]["scaler"].transform(data_features)
                data_transformed = self.model[window_size]["pca"].transform(data_transformed)
                dataframe["label"] = self.model[window_size]["model"].predict(data_transformed)
                for _, row in dataframe.iterrows():
                    start = int(row["frame_start"])
                    for i in range(start, start+window_size):
                        predictions[angle][i].append(int(row["label"]))

        print(f"Voting the results for each angle and frame...")

        for key, val in predictions.items():
            for i, frame in enumerate(predictions[key]):
                if not frame:
                    label = -1
                elif 1 in frame:
                    label = 1
                else:
                    label = 0
                predictions[key][i] = label

        print(f"Done - {strftime('%H:%M:%S', gmtime())}")

        infant = etl.cima[infant_id]

        infant["predictions"] = pd.DataFrame(predictions)

        return infant

if __name__ == "__main__":
    predictor = Predictor()
    predictor.load_model("test_model")
    predictor.predict("/home/login/datasets/CIMA/data", "001")
