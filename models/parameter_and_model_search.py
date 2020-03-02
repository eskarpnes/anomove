import sys
import time
import os
import pandas as pd
from tqdm import tqdm
import multiprocessing as multi
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.abod import ABOD
from etl.etl import ETL
from sklearn import model_selection, neighbors, metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from models import analyse_results as analyse
sys.path.append('../')


def get_search_parameter():
    parameters = {
        "noise_reduction": ["movement"],
        "pooling": ["mean"],
        "bandwidth": [5],
        "pca": [5, 10],
        "window_overlap": [1]
    }
    return parameters

def get_models():
    models = [
        {
            "model": ABOD,
            "fit_x_and_y": False,
            "parameters": {}
        },
        {
            "model": KNN,
            "fit_x_and_y": False,
            "parameters": {
                "n_neighbors": 2
            }
        },
        {
            "model": KNN,
            "fit_x_and_y": False,
            "parameters": {
                "n_neighbors": 5
            }
        },
        {
            "model": LOF,
            "fit_x_and_y": False,
            "parameters": {
                "n_neighbors": 2
            }
        },
        {
            "model": LOF,
            "fit_x_and_y": False,
            "parameters": {
                "n_neighbors": 5
            }
        },
        ]
    return models

def model_testing(data, model):
    X_train, X_test, y_train, y_test = data

    clf = model["model"](**model["parameters"])

    if model["fit_x_and_y"]:
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
        columns=["model", "model_parameter", "noise_reduction", "bandwidth", "pooling", "window_overlap", "pca", "window_size", "angle",
                 "sensitivity", "specificity"]
    )

    # print(f"{len(grid)} different combinations of parameters will be explored.")

    pbar = tqdm(total=(len(grid)*len(models)*len(window_sizes)*len(angles)))

    for i, params in enumerate(grid):
        etl = ETL(
            data_path=DATA_PATH,
            window_sizes=window_sizes,
            bandwidth=params["bandwidth"],
            pooling=params["pooling"],
            noise_reduction=params["noise_reduction"],
            size=size
        )
        etl.load("CIMA")
        etl.preprocess_pooled()
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
                        "bandwidth": params["bandwidth"],
                        "pooling": params["pooling"],
                        "window_overlap": params["window_overlap"],
                        "pca": params["pca"],
                        "window_size": str(window_size),
                        "angle": angle,
                        "sensitivity": sensitivity,
                        "specificity": specificity
                    }, ignore_index=True)
                    pbar.update()
    results.to_csv("model_search_results.csv")
    pbar.close()

if __name__ == '__main__':
    DATA_PATH = "/home/login/datasets"
    window_sizes = [128, 256, 512]
    angles = ["shoulder", "elbow", "hip", "knee"]

    # multi.freeze_support()
    run_search(DATA_PATH, window_sizes, angles, size=64)
    analyse.print_results("model_search_results.csv")

