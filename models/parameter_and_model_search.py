import sys
import time
import pandas as pd
import multiprocessing as multi
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.xgbod import XGBOD
from etl.etl import ETL
from sklearn import model_selection, neighbors, metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from models import analyse_results as ana
sys.path.append('../')


def get_search_parameter():
    parameters = {
        # "noise_reduction": ["movement", "short_vector", "all"],
        # "pooling": ["mean", "max"],
        # "bandwidth": [None, 5, 10],
        # "pca": [None, 5, 10],
        "noise_reduction": ["movement"],
        "pooling": ["mean"],
        "bandwidth": [5],
        "pca": [10],
        "models": [
            {
                "model": KNN,
                "fit_x_and_y": False,
                "parameters":
                    {
                        "n_neighbors": 2,
                    }
            },
            {
                "model": LOF,
                "fit_x_and_y": False,
                "parameters":
                    {
                        "n_neighbors": 2,
                    }
            },
            {
                "model": XGBOD,
                "fit_x_and_y": True,
                "parameters":
                    {
                        "random_state": 42,
                    }
            },
        ]
    }
    return parameters


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


def run_search(path, window_sizes, angles, tiny_data=False):
    DATA_PATH = path
    grid = model_selection.ParameterGrid(get_search_parameter())

    results = pd.DataFrame(
        columns=["model", "model_parameter", "noise_reduction", "bandwidth", "pooling", "pca", "window_size", "angle",
                 "sensitivity", "specificity"]
    )

    print(f"{len(grid)} different combinations of parameters will be explored.")

    for i, params in enumerate(grid):
        etl = ETL(
            data_path=DATA_PATH,
            window_sizes=window_sizes,
            bandwidth=params["bandwidth"],
            pooling=params["pooling"],
            noise_reduction=params["noise_reduction"]
        )
        etl.load("CIMA_angles_resampled_cleaned", tiny=tiny_data)
        etl.generate_fourier_dataset()

        for window_size in window_sizes:
            for angle in angles:

                FOURIER_PATH = DATA_PATH + str(window_size) + "/" + angle + ".json"

                df = pd.read_json(FOURIER_PATH)
                df_features = pd.DataFrame(df.data.tolist())

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

                    model = params["models"]
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
                    "pca": params["pca"],
                    "window_size": str(window_size),
                    "angle": angle,
                    "sensitivity": sensitivity,
                    "specificity": specificity
                }, ignore_index=True)

        # Short rest to prevent race conditions.
        time.sleep(1)
        print(f"{i+1} of {len(grid)} runs done.")

    results.to_csv("model_search_results.csv")


if __name__ == '__main__':
    DATA_PATH = "C://Users//haavalo//Desktop//Master//Dataset//"
    window_sizes = [128]
    angles = ["V3"]

    multi.freeze_support()
    run_search(DATA_PATH, window_sizes, angles, tiny_data=False)
    ana.print_results("model_search_results.csv")

