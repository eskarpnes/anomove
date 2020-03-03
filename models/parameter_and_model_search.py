import sys
sys.path.append('../')
import time
import os
import pandas as pd
from tqdm import tqdm
import multiprocessing as multi
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.abod import ABOD
from pyod.models.sod import SOD
from pyod.models.ocsvm import OCSVM
from pyod.models.hbos import HBOS
from pyod.models.cof import COF
from pyod.models.cblof import CBLOF
from etl.etl import ETL
from sklearn import model_selection, neighbors, metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from models import analyse_results as analyse


def get_search_parameter():
    parameters = {
        "noise_reduction": ["movement"],
        "minimal_movement": [0.02, 0.04, 0.1],
        "pooling": ["mean", "max"],
        "sma": [3, 5],
        "bandwidth": [None, 3, 5],
        "pca": [None, 2, 5, 10],
        "window_overlap": [1, 4, 8]
    }
    return parameters

def get_models():
    models = [
        {
            "model": ABOD,
            "supervised": False,
            "parameters": {}
        },
        {
            "model": KNN,
            "supervised": False,
            "parameters": {
                "n_neighbors": 2
            }
        },
        {
            "model": KNN,
            "supervised": False,
            "parameters": {
                "n_neighbors": 5
            }
        },
        {
            "model": LOF,
            "supervised": False,
            "parameters": {
                "n_neighbors": 2
            }
        },
        {
            "model": LOF,
            "supervised": False,
            "parameters": {
                "n_neighbors": 5
            }
        },
        {
            "model": SOD,
            "supervised": False,
            "parameters": {
                "n_neighbors": 10
            }
        },
        {
            "model": OCSVM,
            "supervised": False,
            "parameters": {}
        },
        {
            "model": HBOS,
            "supervised": False,
            "parameters": {}
        },
        {
            "model": CBLOF,
            "supervised": False,
            "parameters": {}
        },
        ]
    return models

def model_testing(data, model):
    X_train, X_test, y_train, y_test = data

    clf = model["model"](**model["parameters"])

    if model["supervised"]:
        clf.fit(X_train, y_train)
    else:
        clf.fit(X_train)

    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_test_pred, labels=[0, 1]).ravel()

    if tp + fn == 0:
        sensitivity = 0
    else:
        sensitivity = tp / (tp + fn)

    if tn + fp == 0:
        specificity = 0
    else:
        specificity = tn / (tn + fp)

    return sensitivity, specificity


def run_search(path, window_sizes, angles, size=0):
    DATA_PATH = path
    grid = model_selection.ParameterGrid(get_search_parameter())
    models = get_models()

    results = pd.DataFrame(
        columns=["model", "model_parameter", "noise_reduction", "minimal_movement", "bandwidth", "pooling", "sma", "window_overlap", "pca", "window_size", "angle",
                 "sensitivity", "specificity"]
    )

    pbar = tqdm(total=len(grid))

    for i, params in enumerate(grid):
        print(f"Running with params: \n{params}")
        etl = ETL(
            data_path=DATA_PATH,
            window_sizes=window_sizes,
            bandwidth=params["bandwidth"],
            pooling=params["pooling"],
            sma_window=params["sma"],
            noise_reduction=params["noise_reduction"],
            minimal_movement=params["minimal_movement"],
            size=size
        )
        etl.load("CIMA")
        print("\nPreprocessing data.")
        etl.preprocess_pooled()
        print("\nGenerating fourier data.")
        etl.generate_fourier_dataset(window_overlap=params["window_overlap"])

        for window_size in window_sizes:
            for angle in angles:

                RIGHT_FOURIER_PATH = os.path.join(DATA_PATH, str(window_size), "right_" + angle + ".json")
                LEFT_FOURIER_PATH = os.path.join(DATA_PATH, str(window_size), "left_" + angle + ".json")

                right_df = pd.read_json(RIGHT_FOURIER_PATH)
                left_df = pd.read_json(LEFT_FOURIER_PATH)
                df = right_df.append(left_df)
                df_features = pd.DataFrame(df.data.tolist())

                for model in models:
                    print(f"\n Testing model {model['model']} at time {time.strftime('%H:%M:%S',  time.gmtime())}")
                    try:
                        if params["pca"] is not None:
                            pca = PCA(n_components=params["pca"])
                            df_features = StandardScaler().fit_transform(df_features)
                            principal_components = pca.fit_transform(df_features)
                            principal_df = pd.DataFrame(data=principal_components)
                            data = principal_df
                        else:
                            data = df_features
                        labels = df["label"]

                        model_data = model_selection.train_test_split(data, labels, test_size=0.25)
                        sensitivity, specificity = model_testing(model_data, model)

                    except ValueError:
                        sensitivity, specificity = ("crashed", "crashed")

                    results = results.append({
                        "model": model["model"],
                        "model_parameter": model["parameters"],
                        "noise_reduction": params["noise_reduction"],
                        "minimal_movement": params["minimal_movement"],
                        "bandwidth": params["bandwidth"],
                        "pooling": params["pooling"],
                        "sma": params["sma"],
                        "window_overlap": params["window_overlap"],
                        "pca": params["pca"],
                        "window_size": str(window_size),
                        "angle": angle,
                        "sensitivity": sensitivity,
                        "specificity": specificity
                    }, ignore_index=True)
        pbar.update()
        print("\nCheckpoint created.")
        results.to_csv("model_search_results.csv")
    pbar.close()

if __name__ == '__main__':
    DATA_PATH = "/home/erlend/datasets"
    window_sizes = [128, 256, 512, 1024]
    angles = ["shoulder", "elbow", "hip", "knee"]

    # multi.freeze_support()
    run_search(DATA_PATH, window_sizes, angles)
    analyse.print_results("model_search_results.csv")

