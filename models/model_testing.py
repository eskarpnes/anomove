from sklearn import model_selection, neighbors, metrics
import pandas as pd
import numpy as np
from pyod.models.sod import SOD
from pyod.models.hbos import HBOS
from pyod.models.xgbod import XGBOD
from pyod.models.loci import LOCI
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from tqdm import tqdm


def get_data(window_size, angle):
    DATA_PATH = "C://Users//haavalo//Desktop//Master//Dataset//" + str(window_size) + "/" + angle + ".json"

    df = pd.read_json(DATA_PATH)
    data = pd.DataFrame(list(df["data"]))

    X_train, X_test, y_train, y_test = model_selection.train_test_split(data, df["label"], test_size=0.25)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test


def model_testing(model, window_size=128, angle="V3"):
    X_train, X_test, y_train, y_test = get_data(window_size, angle)

    clf_name = model["model_name"]
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

    return [clf_name, window_size, angle, sensitivity, specificity]


if __name__ == '__main__':
    # windows = [128, 256, 512, 1024]
    # angles = ["V3", "V4", "V5", "V6"]
    windows = [128]
    angles = ["V3"]
    results = pd.DataFrame(
        columns=[
            "model_name",
            "parameters",
            "window_size",
            "angle",
            "sensitivity",
            "specificity"]
    )

    # Insert new models in the following template:
    #     "model_name": name of model,
    #     "model": the model,
    #     "fit_x_and_y": if the model needs both x_train and y_train, set this to True, else False,
    #     "parameters":
    #     {
    #         "param1": value,
    #         "param2": value
    #     }
    models = [
        {
            "model_name": "KNN",
            "model": KNN,
            "fit_x_and_y": False,
            "parameters":
                {
                    "n_neighbors": 2,
                    "leaf_size": 30
                }
        },
        # {
        #     "model_name": "XGBOD",
        #     "model": XGBOD,
        #     "fit_x_and_y": True,
        #     "parameters":
        #         {
        #             "random_state": 42
        #         }
        # },
    ]

    for model in tqdm(models):
        for window in tqdm(windows):
            for angle in angles:
                clf_name, result_window_size, result_angle, sensitivity, specificity = model_testing(model, window_size=window, angle=angle)
                results = results.append(
                    {
                        "model_name": clf_name,
                        "parameters": model["parameters"],
                        "window_size": result_window_size,
                        "angle": result_angle,
                        "sensitivity": sensitivity,
                        "specificity": specificity
                    }, ignore_index=True
                )


    print(results)
    results.to_csv("results.csv")