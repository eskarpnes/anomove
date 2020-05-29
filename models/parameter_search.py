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
from sklearn.model_selection import StratifiedKFold
import models.create_models as create_models


def get_search_parameter():
    parameters = {
        "minimal_movement": [0.5, 0.75],
        "sma": [3],
        "pls": [3, 5, 10],
        "window_overlap": [1, 2, 4]
    }
    return parameters


def model_testing(data, model):
    X_train, X_test, y_train, y_test = data

    clf = model["model"](**model["parameters"])

    import time
    start_time = time.time()

    if model["supervised"]:
        clf.fit(X_train, y_train)
    else:
        clf.fit(X_train)

    end_time = time.time()
    elapsed = int(end_time-start_time)

    # print(f"\n{model['model']} - {str(model['parameters'])[:100]}")
    # print(f"elapsed - {elapsed}s\n")

    y_test_pred = clf.predict(X_test) # outlier labels (0 or 1)
    if -1 in y_test_pred:
        for i in range(len(y_test_pred)):
            if y_test_pred[i] == 1:
                y_test_pred[i] = 0
            elif y_test_pred[i] == -1:
                y_test_pred[i] = 1
    # y_test_scores = clf.decision_function(X_test)  # outlier scores

    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_test_pred, labels=[0, 1]).ravel()

    if tp == 0:
        sensitivity = 0
    else:
        sensitivity = tp / (tp + fn)

    if tn == 0:
        specificity = 0
    else:
        specificity = tn / (tn + fp)

    y_test_scores = clf.predict_proba(X_test)
    if len(y_test_scores.shape) != 1:
        y_test_scores = y_test_scores[:,1]

    roc_auc = metrics.roc_auc_score(y_test, y_test_scores)

    return sensitivity, specificity, roc_auc


def chunkify(large_list, chunk_size):
    for i in range(0, len(large_list), chunk_size):
        yield large_list[i:i + chunk_size]


def run_search(path, window_sizes, angles, models, size=0, result_name="search_results", novelty=False, kfold_splits=5):
    DATA_PATH = path
    grid = model_selection.ParameterGrid(get_search_parameter())

    results = pd.DataFrame(
        columns=["model", "model_parameter", "minimal_movement", "sma",
                 "window_overlap", "pls", "window_size", "angle",
                 "sensitivity", "specificity"]
    )

    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    for i, params in enumerate(grid):

        print(f"Running with params: \n{params}")

        if run_done(i, len(window_sizes) * len(angles)):
            print(f"Found parameters in checkpoints, skipping...")
            continue

        generate_fourier(DATA_PATH, window_sizes, size, params)

        # print(f"The number of methods without k-folding are: {str(len(models))}")
        pbar = tqdm(total=len(models) * len(window_sizes) * len(angles) * kfold_splits)

        def update_progress(*a):
            pbar.update()

        for window_size, angle in iterate_angles():

            if result_exist(result_name, i, window_size, angle):
                print(f"Found this combination in checkpoints, skipping...")
                pbar.update(len(models)*kfold_splits)
                continue

            with Manager() as manager:
                synced_results = manager.list()

                data, labels = load_fourier_angle(window_size, angle)

                data_amount = data.shape[0]

                print(f"Data amount:  {data_amount}")

                for batch in chunkify(models, 1):
                    pool = Pool()

                    kfold_parameters = {
                        "batch": batch,
                        "pool": pool,
                        "angle": angle,
                        "splits": kfold_splits,
                        "pls_components": params["pls"],
                        "novelty": novelty
                    }

                    async_kfold(data, labels, kfold_parameters, synced_results, update_progress)

                    pool.close()
                    pool.join()

                print("\nCheckpoint created.")
                checkpoint_name = f"{result_name.split('/')[-1]}_{str(window_size)}_{angle}_{i}.csv"
                dump_results(params, synced_results, window_size, checkpoint_name)
        pbar.close()
    save_and_clean(result_name)

def result_exist(result_name, i, window_size, angle):
    checkpoint_name = f"{result_name.split('/')[-1]}_{str(window_size)}_{angle}_{i}.csv"
    return os.path.exists(f"tmp/{checkpoint_name}")

def run_done(run_num, runs):
    filenames = os.listdir("tmp")
    suffix = f"_{run_num}.csv"
    done = [run for run in filenames if suffix in run]
    return len(done) == runs


def generate_fourier(data_path, window_sizes, size, params):
    etl = ETL(
        data_path=data_path,
        window_sizes=window_sizes,
        sma_window=params["sma"],
        minimal_movement=params["minimal_movement"],
        size=size
    )
    etl.load("CIMA")
    print("\nPreprocessing data.")
    etl.preprocess_pooled()
    print("\nGenerating fourier data.")
    etl.generate_fourier_dataset(window_overlap=params["window_overlap"])

def save_and_clean(result_name):
    final_results = pd.DataFrame()
    for filename in os.listdir("tmp"):
        sub_results = pd.read_csv(f"tmp/{filename}")
        final_results = final_results.append(sub_results)
    final_results.to_csv(f"{result_name}.csv", index=False)
    print("Do you want to delete the temporary files?")
    answer = input("y or n: ")
    if answer.lower() == "y":
        shutil.rmtree("tmp")

def iterate_angles():
    for window_size in window_sizes:
        for angle in angles:
            yield window_size, angle

def load_fourier_angle(window_size, angle):
    right_fourier_path = os.path.join(DATA_PATH, str(window_size), "right_" + angle + ".json")
    left_fourier_path = os.path.join(DATA_PATH, str(window_size), "left_" + angle + ".json")

    right_df = pd.read_json(right_fourier_path)
    left_df = pd.read_json(left_fourier_path)
    df = right_df.append(left_df)
    df.reset_index(drop=True, inplace=True)

    data = pd.DataFrame(df.data.tolist())
    labels = df["label"]

    return data, labels


def dump_results(params, synced_results, window_size, path):
    results = []
    for result in synced_results:
        results.append({
            "model": result["model"],
            "model_parameter": result["parameters"],
            "minimal_movement": params["minimal_movement"],
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
    result_path = os.path.join("tmp", path)
    results.to_csv(result_path)

def async_kfold(X, y, parameters, synced_results, callback):
    kf = StratifiedKFold(n_splits=parameters["splits"])
    for train_index, test_index in kf.split(X, y):
        x_train = X.iloc[train_index]
        y_train = y[train_index]
        if parameters["novelty"]:
            x_train = x_train.loc[y_train == 0]
            y_train = y_train.loc[y_train == 0]
        x_test = X.iloc[test_index]
        y_test = y[test_index]

        if parameters["pls_components"] != 0:
            pls = PLSRegression(n_components=parameters["pls_components"])
            pls.fit(x_train, y_train)
            x_train = pls.transform(x_train)
            x_test = pls.transform(x_test)

        model_data = x_train, x_test, y_train, y_test
        for model in parameters["batch"]:
            parameters["pool"].apply_async(async_model_testing, args=(model_data, model, synced_results, parameters["angle"],),
                             callback=callback)
            # async_model_testing(model_data, model, synced_results, parameters["angle"])

def async_model_testing(model_data, model, synced_result, angle):
    try:
        # print(f"Started fitting {model['model']}")
        sensitivity, specificity, roc_auc = model_testing(model_data, model)
    except Exception as e:
        print("Unexpected error:", sys.exc_info()[0])
        print(e)
        # print(model["parameters"])
        print(f"{model['model']} crashed.")

        sensitivity, specificity, roc_auc = ("crashed", "crashed", "crashed")
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
        "minimal_movement",
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
    from pyod.models.cblof import CBLOF
    from pyod.models.hbos import HBOS
    from pyod.models.iforest import IForest
    from pyod.models.abod import ABOD
    from pyod.models.ocsvm import OCSVM

    estimator_list = []

    # predefined range of n_neighbors for KNN, AvgKNN, and LOF
    k_range = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


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
        if k <= 20:
            estimator_list.append({
                "model": ABOD,
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

    # Cluster range

    n_clusters_range = [8, 10, 12, 14, 16, 18, 20]
    for n_clusters in n_clusters_range:
        estimator_list.append({
            "model": CBLOF,
            "supervised": False,
            "parameters": {
                "n_clusters": n_clusters,
                "contamination": 0.05
            }
        })

    return estimator_list

def construct_raw_base_estimators():
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF
    from pyod.models.cblof import CBLOF
    from pyod.models.hbos import HBOS
    from pyod.models.iforest import IForest
    from pyod.models.abod import ABOD
    from pyod.models.ocsvm import OCSVM

    estimator_list = []

    # predefined range of n_neighbors for KNN, AvgKNN, and LOF
    k_range = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for k in k_range:
        estimator_list.append(KNN(n_neighbors=k, method="largest", contamination=0.05))
        estimator_list.append(KNN(n_neighbors=k, method="mean", contamination=0.05))
        estimator_list.append(LOF(n_neighbors=k, contamination=0.05))
        # if k <= 20:
        #     estimator_list.append(ABOD(n_neighbors=k, contamination=0.05))

    # n_bins_range = [3, 5, 7, 9, 12, 15, 20, 25, 30, 50]
    # for n_bins in n_bins_range:
    #     estimator_list.append(HBOS(n_bins=n_bins, contamination=0.05))

    # predefined range of nu for one-class svm
    nu_range = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    for nu in nu_range:
        estimator_list.append(OCSVM(nu=nu, contamination=0.05))

    # predefined range for number of estimators in isolation forests
    n_range = [10, 20, 50, 70, 100, 150, 200, 250]
    for n in n_range:
        estimator_list.append(IForest(n_estimators=n, random_state=42, contamination=0.05))

    # Cluster range

    # n_clusters_range = [8, 10, 12, 14, 16, 18, 20]
    # for n_clusters in n_clusters_range:
    #     estimator_list.append(CBLOF(n_clusters=n_clusters, contamination=0.05))

    return estimator_list

def construct_xgbod():
    from pyod.models.xgbod import XGBOD
    model = {
        "model": XGBOD,
        "supervised": True,
        "parameters": {
            "estimator_list": construct_raw_base_estimators(),
            "silent": False,
            "n_jobs": 6
        }
    }
    return [model]

def construct_lscp():
    from pyod.models.lscp import LSCP
    model = {
        "model": LSCP,
        "supervised": False,
        "parameters": {
            "detector_list": construct_raw_base_estimators(),
            "local_region_size": 150,
            "n_bins": 10,
            "random_state": 42
        }
    }
    return [model]

def construct_simple_aggregator():
    from combo.models.detector_comb import SimpleDetectorAggregator
    models = []
    models.append({
        "model": SimpleDetectorAggregator,
        "supervised": False,
        "parameters": {
            "base_estimators": construct_raw_base_estimators(),
            "method": "average"
        }
    })
    models.append({
        "model": SimpleDetectorAggregator,
        "supervised": False,
        "parameters": {
            "base_estimators": construct_raw_base_estimators(),
            "method": "maximization"
        }
    })
    return models

if __name__ == '__main__':
    DATA_PATH = "/home/erlend/datasets"
    window_sizes = [128, 256, 512, 1024]
    angles = ["shoulder", "elbow", "hip", "knee"]
    base_estimators = construct_base_estimators()
    aggregators = construct_simple_aggregator()
    xgbod = construct_xgbod()
    lscp = construct_lscp()

    models = []

    for aggregator in aggregators:
        models.append(aggregator)

    for model in xgbod:
        models.append(model)

    for model in lscp:
        models.append(model)

    run_search(DATA_PATH, window_sizes, angles, models, result_name="results/ensembles")

    # average_results("results/ensembles.csv")

    # from analyse_results import print_results
    # print_results("results/xgbod.csv", sort_by=["roc_auc"])
