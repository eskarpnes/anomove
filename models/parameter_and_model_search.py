import sys
sys.path.append('../')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import freeze_support, Pool, Manager
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
from sklearn.model_selection import KFold
from models import create_models


def get_search_parameter():
    parameters = {
        "noise_reduction": ["movement"],
        "minimal_movement": [0.1],
        "pooling": ["mean"],
        "sma": [3],
        "bandwidth": [5],
        "pca": [10],
        "window_overlap": [1]
    }
    return parameters


def get_models():
    return create_models.create_models()


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
    if os.path.exists("model_search_results.csv"):
        results = pd.read_csv("model_search_results.csv", index_col=0)
    else:
        results = pd.DataFrame(
            columns=["model", "model_parameter", "noise_reduction", "minimal_movement", "bandwidth", "pooling", "sma", "window_overlap", "pca", "window_size", "angle",
                     "sensitivity", "specificity"]
        )

    pbar = tqdm(total=len(grid))

    kf = KFold(n_splits=10)
    for i, params in enumerate(grid):

        print(f"Running with params: \n{params}")

        params_series = pd.Series(params)
        params_keys = list(params.keys())
        in_results = (results[params_keys] == params_series).all(axis=1).sum()
        if in_results:
            # Parameters has already been ran.
            print("Already done this combination, skipping...\n")
            pbar.update()
            continue

        if params["pca"] == 0 and params["bandwidth"] == 0:
            # No dimension reduction, too many dimensions.
            print("No dimension reduction, skipping...\n")
            pbar.update()
            continue

        if params["bandwidth"] == 0 and params["pooling"] == "max":
            # Pooling is dependent on bandwidth, so no bandwidth = no pooling. Remove one choice in pooling to reduce it to 1.
            print("Invalid combination, skipping...\n")
            pbar.update()
            continue

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
            with Manager() as manager:
                print("Starting a pool of model fitting.")
                synced_results = manager.list()
                pool = Pool()

                for angle in angles:
                    RIGHT_FOURIER_PATH = os.path.join(DATA_PATH, str(window_size), "right_" + angle + ".json")
                    LEFT_FOURIER_PATH = os.path.join(DATA_PATH, str(window_size), "left_" + angle + ".json")

                    right_df = pd.read_json(RIGHT_FOURIER_PATH)
                    left_df = pd.read_json(LEFT_FOURIER_PATH)
                    df = right_df.append(left_df)
                    df.reset_index(drop=True, inplace=True)
                    df_features = pd.DataFrame(df.data.tolist())
                    for model in models:
                        if params["pca"] != 0:
                            pca = PCA(n_components=params["pca"])
                            df_features = StandardScaler().fit_transform(df_features)
                            principal_components = pca.fit_transform(df_features)
                            principal_df = pd.DataFrame(data=principal_components)
                            data = principal_df
                        else:
                            data = df_features
                        labels = df["label"]

                        for train_index, test_index in kf.split(data):
                            x_train = data.iloc[train_index]
                            x_test = data.iloc[test_index]
                            y_train = labels[train_index]
                            y_test = labels[test_index]

                            model_data = x_train, x_test, y_train, y_test
                            pool.apply_async(async_model_testing, args=(model_data, model, synced_results, angle,))

                pool.close()
                pool.join()

                for result in synced_results:
                    results = results.append({
                        "model": result["model"],
                        "model_parameter": result["parameters"],
                        "noise_reduction": params["noise_reduction"],
                        "minimal_movement": params["minimal_movement"],
                        "bandwidth": params["bandwidth"],
                        "pooling": params["pooling"],
                        "sma": params["sma"],
                        "window_overlap": params["window_overlap"],
                        "pca": params["pca"],
                        "window_size": str(window_size),
                        "angle": result["angle"],
                        "sensitivity": result["sensitivity"],
                        "specificity": result["specificity"]
                    }, ignore_index=True)
        pbar.update()
        print("\nCheckpoint created.")
        results.to_csv("model_search_results.csv")
    pbar.close()


def async_model_testing(model_data, model, synced_result, angle):
    try:
        # print(f"Started fitting {model['model']}")
        sensitivity, specificity = model_testing(model_data, model)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        print(f"{model['model']} crashed.")

        sensitivity, specificity = ("crashed", "crashed")
    synced_result.append({
        "model": model["model"],
        "parameters": model["parameters"],
        "angle": angle,
        "sensitivity": sensitivity,
        "specificity": specificity
    })


def kf_results():
    results = pd.read_csv("model_search_results_testing.csv", index_col=0)
    for i, row in results.iterrows():
        counter = 0
        for j, row2 in results.iterrows():
            if row.iloc[0:11].equals(row2.iloc[0:11]):
                counter += 1
                if counter > 1:
                    print("True")
        # gå gjennom dataframen og finn alle like modeller
        # regn så ut variansen og avg_senv og avg_spec
        # slett så radene utenom den nye raden med avg_sens og avg_spec
        # print(index)



if __name__ == '__main__':
    DATA_PATH = "/home/erlend/datasets"
    window_sizes = [128, 256, 512, 1024]
    angles = ["shoulder", "elbow", "hip", "knee"]

    # freeze_support()
    run_search(DATA_PATH, window_sizes, angles)
    # analyse.print_results("model_search_results.csv")