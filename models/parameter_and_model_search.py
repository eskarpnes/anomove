import sys
import gc
sys.path.append('../')
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import shutil
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import freeze_support, Pool, Manager, cpu_count
from etl.etl import ETL
from sklearn import model_selection, neighbors, metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from models.create_models import get_models, create_tunable_ensemble


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


def chunkify(large_list, chunk_size):
    for i in range(0, len(large_list), chunk_size):
        yield large_list[i:i + chunk_size]


def run_search(path, window_sizes, angles, size=0, ensemble=False, result_name="search_results"):
    DATA_PATH = path
    grid = model_selection.ParameterGrid(get_search_parameter())
    # Returns base models
    #models = get_models(ensemble=ensemble, knn_methods=["mean", "largest"], pca=10)
    # Returns ensemble models
    # models = get_models(ensemble=True, knn_methods=["mean", "largest"], ensemble_combinations=["average", "maximization"], pca=10)
    # Returns ensemble with only LOF
    # models = get_models(ensemble=True, ensemble_combinations=["average"], pca=10, only_LOF=True)
    # Returns tunable neighbor parameter ensemble
    knn_neighbors = [5, 9, 10]
    lof_neighbors = [6, 7, 8, 9, 10]
    abod_neighbors = [3, 4, 5, 6]
    models = create_tunable_ensemble(knn_neighbors, lof_neighbors, abod_neighbors)
    kfold_splits = 5
    kf = KFold(n_splits=kfold_splits)

    if os.path.exists("model_search_results.csv"):
        results = pd.read_csv("model_search_results.csv", index_col=0)
    else:
        results = pd.DataFrame(
            columns=["model", "model_parameter", "noise_reduction", "minimal_movement", "bandwidth", "pooling", "sma",
                     "window_overlap", "pca", "window_size", "angle",
                     "sensitivity", "specificity"]
        )

    print(f"The number of methods without k-folding are: {str(len(models))}")
    pbar = tqdm(total=len(models) * len(window_sizes) * len(angles) * kfold_splits)

    def update_progress(*a):
        pbar.update()

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

        if not os.path.exists("tmp"):
            os.mkdir("tmp")

        for window_size in window_sizes:
            for angle in angles:
                with Manager() as manager:
                    synced_results = manager.list()

                    right_fourier_path = os.path.join(DATA_PATH, str(window_size), "right_" + angle + ".json")
                    left_fourier_path = os.path.join(DATA_PATH, str(window_size), "left_" + angle + ".json")

                    right_df = pd.read_json(right_fourier_path)
                    left_df = pd.read_json(left_fourier_path)
                    df = right_df.append(left_df)
                    df.reset_index(drop=True, inplace=True)
                    df_features = pd.DataFrame(df.data.tolist())

                    for batch in chunkify(models, 1):
                        pool = Pool()

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
                            for model in batch:
                                pool.apply_async(async_model_testing, args=(model_data, model, synced_results, angle,),
                                                 callback=update_progress)

                        pool.close()
                        pool.join()


                    print("\nCheckpoint created.")
                    results = []
                    for result in synced_results:
                        results.append({
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
                        })
                    results = pd.DataFrame(results)
                    result_path = os.path.join("tmp", f"{result_name}_{str(window_size)}_{angle}.csv")
                    results.to_csv(result_path)
                    del results
                gc.collect()

    final_results = pd.DataFrame()
    for filename in os.listdir("tmp"):
        sub_results = pd.read_csv(f"tmp/{filename}")
        final_results = final_results.append(sub_results)
    final_results.to_csv(f"{result_name}.csv")
    shutil.rmtree("tmp")
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


def average_results(file):
    results = pd.read_csv(file, index_col=0, engine="python")

    results = results.groupby([
        "model",
        "model_parameter",
        "noise_reduction",
        "minimal_movement",
        "bandwidth",
        "pooling",
        "sma",
        "window_overlap",
        "pca",
        "window_size",
        "angle"
    ]).mean().reset_index()
    results.to_csv(file[0:-4] + "_groupBy.csv")


if __name__ == '__main__':
    DATA_PATH = "/home/erlend/datasets"
    window_sizes = [128, 256, 512, 1024]
    angles = ["shoulder", "elbow", "hip", "knee"]

    # freeze_support()
    # run_search(DATA_PATH, window_sizes, angles, ensemble=False, result_name="model_search_kfold")
    # run_search(DATA_PATH, window_sizes, angles, ensemble=True, result_name="ensemble_search_kfold")
    average_results("results//model_abod_search_kfold.csv")
