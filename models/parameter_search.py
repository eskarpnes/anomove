import sys
import gc

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM

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
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from pyod.models.xgbod import XGBOD
import models.create_models as create_models


def get_search_parameter():
    parameters = {
        "noise_reduction": ["movement"],
        "minimal_movement": [0.5],
        "pooling": ["mean"],
        "sma": [3],
        "bandwidth": [0],
        "pls": [5],
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

    # y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    # y_train_scores = clf.decision_scores_  # raw outlier scores

    y_test_pred = clf.predict(X_test) # outlier labels (0 or 1)
    if -1 in y_test_pred:
        for i in range(len(y_test_pred)):
            if y_test_pred[i] == 1:
                y_test_pred[i] = 0
            elif y_test_pred[i] == -1:
                y_test_pred[i] = 1
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_test_pred, labels=[0, 1]).ravel()

    if tp == 0:
        sensitivity = 0
    else:
        sensitivity = tp / (tp + fn)

    if tn == 0:
        specificity = 0
    else:
        specificity = tn / (tn + fp)

    roc_auc = metrics.roc_auc_score(y_test, y_test_pred)

    return sensitivity, specificity, roc_auc


def chunkify(large_list, chunk_size):
    for i in range(0, len(large_list), chunk_size):
        yield large_list[i:i + chunk_size]


def run_search(path, window_sizes, angles, models, size=0, result_name="search_results", novelty=False):
    DATA_PATH = path
    grid = model_selection.ParameterGrid(get_search_parameter())
    kfold_splits = 5
    kf = KFold(n_splits=kfold_splits)

    if os.path.exists("model_search_results.csv"):
        results = pd.read_csv("model_search_results.csv", index_col=0)
    else:
        results = pd.DataFrame(
            columns=["model", "model_parameter", "noise_reduction", "minimal_movement", "bandwidth", "pooling", "sma",
                     "window_overlap", "pls", "window_size", "angle",
                     "sensitivity", "specificity"]
        )

    for i, params in enumerate(grid):

        print(f"Running with params: \n{params}")

        params_series = pd.Series(params)
        params_keys = list(params.keys())

        if params["pls"] == 0 and params["bandwidth"] == 0:
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

        print(f"The number of methods without k-folding are: {str(len(models))}")
        pbar = tqdm(total=len(models) * len(window_sizes) * len(angles) * kfold_splits)

        def update_progress(*a):
            pbar.update()

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

                    for batch in chunkify(models, 20):
                        pool = Pool()


                        data = df_features
                        labels = df["label"]

                        for train_index, test_index in kf.split(data):
                            x_train = data.iloc[train_index]
                            y_train = labels[train_index]
                            if novelty:
                                x_train = x_train.loc[y_train == 0]
                                y_train = y_train.loc[y_train == 0]
                            x_test = data.iloc[test_index]
                            y_test = labels[test_index]

                            if params["pls"] != 0:
                                pls = PLSRegression(n_components=params["pls"])
                                pls.fit(x_train, y_train)
                                x_train = pls.transform(x_train)
                                x_test = pls.transform(x_test)


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
                            "pls": params["pls"],
                            "window_size": str(window_size),
                            "angle": result["angle"],
                            "sensitivity": result["sensitivity"],
                            "specificity": result["specificity"],
                            "roc_auc": result["roc_auc"]
                        })
                    results = pd.DataFrame(results)
                    result_path = os.path.join("tmp", f"{result_name.split('/')[-1]}_{str(window_size)}_{angle}_{i}.csv")
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
        sensitivity, specificity, roc_auc = model_testing(model_data, model)
    except Exception as e:
        print("Unexpected error:", sys.exc_info()[0])
        print(e)
        print(f"{model['model']} crashed.")

        sensitivity, specificity = ("crashed", "crashed")
    synced_result.append({
        "model": model["model"],
        "parameters": model["parameters"],
        "angle": angle,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "roc_auc": roc_auc
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
        "pls",
        "window_size",
        "angle"
    ]).mean().reset_index()
    results.to_csv(file[0:-4] + "_groupBy.csv")


def construct_base_estimators():
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF
    from pyod.models.hbos import HBOS
    from pyod.models.iforest import IForest
    from pyod.models.ocsvm import OCSVM

    estimator_list = []

    # predefined range of n_neighbors for KNN, AvgKNN, and LOF
    k_range = [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # validate the value of k
    k_range = [k for k in k_range]

    for k in k_range:
        estimator_list.append({
            "model": KNN,
            "supervised": False,
            "parameters": {
                "n_neighbors": k,
                "method": 'largest',
                "contamination": 0.05
            }
        })
        estimator_list.append({
            "model": KNN,
            "supervised": False,
            "parameters": {
                "n_neighbors": k,
                "method": 'mean',
                "contamination": 0.05
            }
        })
        estimator_list.append({
            "model": LOF,
            "supervised": False,
            "parameters": {
                "n_neighbors": k,
                "contamination": 0.05
            }
        })

    n_bins_range = [3, 5, 7, 9, 12, 15, 20, 25, 30, 50]
    for n_bins in n_bins_range:
        estimator_list.append({
            "model": HBOS,
            "supervised": False,
            "parameters": {
                "n_bins": n_bins,
                "contamination": 0.05
            }
        })

    # predefined range of nu for one-class svm
    nu_range = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    for nu in nu_range:
        estimator_list.append({
            "model": OCSVM,
            "supervised": False,
            "parameters": {
                "nu": nu,
                "contamination": 0.05
            }
        })

    # predefined range for number of estimators in isolation forests
    n_range = [10, 20, 50, 70, 100, 150, 200, 250]
    for n in n_range:
        estimator_list.append({
            "model": IForest,
            "supervised": False,
            "parameters": {
                "n_estimators": n,
                "random_state": 42,
                "contamination": 0.05
            }
        })

    return estimator_list

def construct_xgbod():

    model = {
            "model": XGBOD,
            "supervised": True,
            "parameters": {
                "silent": False,
                "n_jobs": 6
            }
    }
    return [model]

if __name__ == '__main__':
    DATA_PATH = "/home/erlend/datasets"
    window_sizes = [128, 256, 512, 1024]
    angles = ["shoulder", "elbow", "hip", "knee"]
    models = construct_base_estimators()

    run_search(DATA_PATH, window_sizes, angles, models, result_name="results/base_estimators")

    average_results("results/base_estimators.csv")

    from analyse_results import print_results
    print_results("results/base_estimators_groupBy.csv", sort_by=["roc_auc"])
