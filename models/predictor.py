import pandas as pd
import os
import joblib


class Predictor:

    def __init__(self):
        self.model = None

    def load_model(self, model_name):
        path = os.path.join("saved_models", model_name + ".joblib")
        self.model = joblib.load(path)

    def save_model(self, model_name):
        path = os.path.join("saved_models", model_name + ".joblib")
        joblib.dump(self.model, path)

    def fit_model(self, X):
        pass

    def predict(self, dataframe):
        # Takes a preprocessed dataframe in with raw values, and does the necessary fourier transform to predict each frame of one infant
        pass
