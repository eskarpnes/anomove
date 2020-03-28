import pandas as pd
import joblib
from sklearn import model_selection, neighbors, metrics
from pyod.models.lof import LOF
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from combo.models.detector_comb import SimpleDetectorAggregator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append("../")
from etl.etl import ETL

DATA_PATH = "/home/erlend/datasets"


def load_data(dataset):
    etl = ETL(
            DATA_PATH,
            [128, 256, 512, 1024],
            pooling="mean",
            sma_window=3,
            bandwidth=5,
            minimal_movement=0.05
        )
    etl.load(dataset)
    etl.preprocess_pooled()
    etl.generate_fourier_dataset(window_overlap=8)


def construct_model():
    classifiers = [
        LOF(n_neighbors=1),
        LOF(n_neighbors=3),
        LOF(n_neighbors=5),
        LOF(n_neighbors=7),
        LOF(n_neighbors=8),
        LOF(n_neighbors=9),
        LOF(n_neighbors=10),
        ABOD(n_neighbors=5),
        KNN(n_neighbors=5),
        OCSVM()
    ]
    model = SimpleDetectorAggregator(
        classifiers,
        method="average"
    )
    return model


def train_model(model_name):
    window_sizes = [128, 256, 512, 1024]
    angles = ["right_shoulder", "left_shoulder", "right_elbow", "left_elbow", "right_hip", "left_hip", "right_knee",
              "left_knee"]
    models = {}
    for window_size in window_sizes:
        scaler = StandardScaler()
        pca = PCA(n_components=10)
        X = pd.DataFrame()
        for angle in angles:
            fourier_path = os.path.join(DATA_PATH, str(window_size), angle + ".json")
            df = pd.read_json(fourier_path)
            X = X.append(df)
        X_features = pd.DataFrame(X.data.tolist())
        X_scaled = scaler.fit_transform(X_features)
        X_pca = pca.fit_transform(X_scaled)
        # y = X["label"]

        model = construct_model()
        model.fit(X_pca)

        models[window_size] = {
            "scaler": scaler,
            "pca": pca,
            "model": model
        }

    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")
    save_path = os.path.join("saved_models", model_name + ".joblib")
    joblib.dump(models, save_path)


if __name__ == "__main__":
    load_data("CIMA")
    train_model("ensemble_model_low_movement")
