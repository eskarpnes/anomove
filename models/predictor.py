import pandas as pd
import numpy as np
import os
from time import gmtime, strftime
import joblib
import sys

sys.path.append("../")
from etl.etl import ETL


class Prediction:

    def __init__(self, video_length):
        self.true_label = None
        self.infant_id = None
        self.data = {}
        self.scores = None
        self.cached_window_method = None
        self.cached_window_sizes = None
        self.video_length = video_length
        self.window_sizes = [128, 256, 512, 1024]
        self.angles = ["right_shoulder", "left_shoulder", "right_elbow", "left_elbow",
                       "right_hip", "left_hip", "right_knee", "left_knee"]

    def set_label(self, label):
        self.true_label = label

    def set_id(self, infant_id):
        self.infant_id = infant_id

    def set_window_data(self, window_size, angle, data):
        if window_size not in self.data.keys():
            self.data[window_size] = {}

        self.data[window_size][angle] = data

    def set_window_sizes(self, window_sizes):
        self.window_sizes = window_sizes

    def aggregate_windows(self, method):
        scores = {}
        for angle in self.angles:
            scores[angle] = pd.Series([[] for i in range(self.video_length)])
        for window_size in self.window_sizes:
            for angle in self.angles:
                for _, row in self.data[window_size][angle].iterrows():
                    start = int(row["frame_start"])
                    for i in range(start, start + window_size):
                        scores[angle][i].append(row["label"])
        for key, val in scores.items():
            for i, frame in enumerate(scores[key]):
                label = 0
                if frame:
                    if method == "mean":
                        label = np.mean(frame)
                    elif method == "max":
                        label = np.max(frame)
                scores[key][i] = label
        self.scores = scores
        self.cached_window_method = method
        self.cached_window_sizes = self.window_sizes
        return scores

    def score_threshold(self, window_method, angle_method, threshold):

        if self.cached_window_method == window_method and set(self.cached_window_sizes) == set(self.window_sizes) and self.scores is not None:
            scores = self.scores
        else:
            scores = self.aggregate_windows(window_method)

        threshold_ratios = []

        for key, val in scores.items():
            thresholds = [1 if score > threshold else 0 for score in val]
            if len(scores[key][scores[key] != 0]) != 0:
                threshold_ratios.append(sum(thresholds) / len(scores[key][scores[key] != 0]))

        return threshold_ratios

    def score(self, window_method, angle_method):
        if self.cached_window_method == window_method and set(self.cached_window_sizes) == set(self.window_sizes) and self.scores is not None:
            scores = self.scores
        else:
            scores = self.aggregate_windows(window_method)

        if angle_method == "mean":
            angle_scores = [np.mean(scores[key][scores[key] != 0]) for key, val in scores.items()]
        elif angle_method == "max":
            angle_scores = [np.max(scores[key][scores[key] != 0]) for key, val in scores.items()]

        return angle_scores


class Predictor:

    def __init__(self, verbose=True, method="all"):
        self.model = None
        self.verbose = verbose
        self.method = method

    def load_model(self, model_name):
        path = os.path.join("saved_models", model_name + ".joblib")
        self.model = joblib.load(path)

    def save_model(self, model_name):
        if not os.path.exists("saved_models"):
            os.mkdir("saved_models")
        path = os.path.join("saved_models", model_name + ".joblib")
        joblib.dump(self.model, path)

    def predict(self, data_path, infant_id):
        if self.verbose:
            print(f"Predicting infant {infant_id} - {strftime('%H:%M:%S', gmtime())}")
        window_sizes = [128, 256, 512, 1024]
        etl = ETL(
            data_path,
            window_sizes,
            pooling="mean",
            sma_window=3,
            bandwidth=0,
            minimal_movement=0.75
        )
        etl.load_infant(infant_id)
        if self.verbose:
            print(f"Preprocessing the data - {strftime('%H:%M:%S', gmtime())}")
        etl.preprocess_pooled()

        angles = ["right_shoulder", "left_shoulder", "right_elbow", "left_elbow", "right_hip", "left_hip", "right_knee",
                  "left_knee"]
        predictions = {}
        video_length = len(etl.cima[infant_id]["data"])
        prediction = Prediction(video_length)
        for angle in angles:
            predictions[angle] = pd.Series([[] for i in range(len(etl.cima[infant_id]["data"]))])

        if self.verbose:
            print(f"Generating fourier data - {strftime('%H:%M:%S', gmtime())}")
        for window_size in window_sizes:
            for angle in angles:
                dataframe = etl.generate_fourier_data(angle, window_size, window_size // 4)
                data_features = pd.DataFrame(dataframe.data.tolist())
                if not data_features.empty:
                    data_transformed = self.model[window_size]["pls"].transform(data_features)
                    dataframe["label"] = self.model[window_size]["model"].predict_proba(data_transformed)
                else:
                    dataframe["label"] = pd.Series([])
                prediction.set_window_data(window_size, angle, dataframe)

        infant = etl.cima[infant_id]
        infant["predictions"] = prediction

        return infant, prediction


if __name__ == "__main__":
    predictor = Predictor()
    predictor.load_model("xgbod")
    # 014 is impaired, 001 is healthy
    infants = ["047", "048", "060", "064", "065", "081", "092"]
    for infant in infants:
        print("-------------------------------------------------------------------------------------------------------")
        prediction = predictor.predict("/home/erlend/datasets/CIMA/validation", infant)
        prediction.score_threshold("mean", "mean", 0.2)
