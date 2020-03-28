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
        path = os.path.join("models", "saved_models", model_name + ".joblib")
        self.model = joblib.load(path)

    def save_model(self, model_name):
        if not os.path.exists("saved_models"):
            os.mkdir("saved_models")
        path = os.path.join("saved_models", model_name + ".joblib")
        joblib.dump(self.model, path)

    def predict(self, data_path, infant_id):
        print(f"Predicting infant {infant_id} - {strftime('%H:%M:%S', gmtime())}")
        window_sizes = [128, 256, 512, 1024]
        etl = ETL(
            data_path,
            window_sizes,
            pooling="mean",
            sma_window=3,
            bandwidth=5,
            minimal_movement=0.1
        )
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
                dataframe = etl.generate_fourier_data(window_size, angle, window_size//8)
                data_features = pd.DataFrame(dataframe.data.tolist())
                if not data_features.empty:
                    data_transformed = self.model[window_size]["scaler"].transform(data_features)
                    data_transformed = self.model[window_size]["pca"].transform(data_transformed)
                    dataframe["label"] = self.model[window_size]["model"].predict_proba(data_transformed)
                else:
                    dataframe["label"] = pd.Series([])
                for _, row in dataframe.iterrows():
                    start = int(row["frame_start"])
                    for i in range(start, start+window_size):
                        predictions[angle][i].append(row["label"])

        print(f"Voting the results for each angle and frame...")



        for key, val in predictions.items():
            thresholds = {
                0.1: 0,
                0.2: 0,
                0.3: 0,
                0.4: 0,
                0.5: 0,
                0.6: 0,
                0.7: 0,
                0.8: 0,
                0.9: 0,
                0.95: 0
            }
            for i, frame in enumerate(predictions[key]):
                if not frame:
                    label = 0
                    # continue
                else:
                    label = np.mean(frame)
                    for threshold in thresholds.keys():
                        if label > threshold:
                            thresholds[threshold] += 1
                predictions[key][i] = label
            for threshold, val in thresholds.items():
                try:
                    thresholds[threshold] = val/len(predictions[key][predictions[key] != 0])
                except ZeroDivisionError:
                    thresholds[threshold] = 0
            print(f"{key} - {np.mean(predictions[key][predictions[key] != 0])}")
            print(f"{key} - {thresholds}")
        print(f"Done - {strftime('%H:%M:%S', gmtime())}")
        print(f"Final results: {np.mean([np.mean(predictions[key]) for key in predictions.keys()])}")
        infant = etl.cima[infant_id]

        infant["predictions"] = pd.DataFrame(predictions)

        return infant


if __name__ == "__main__":
    predictor = Predictor()
    predictor.load_model("ensemble_model")
    # 014 is impaired, 001 is healthy
    infants = ["001", "002", "003", "014", "016", "019"]
    for infant in infants:
        print("-------------------------------------------------------------------------------------------------------")
        predictor.predict("/home/erlend/datasets/CIMA/data", infant)
