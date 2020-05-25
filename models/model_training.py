import pandas as pd
import joblib
from sklearn import model_selection, neighbors, metrics
from pyod.models.lof import LOF
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from combo.models.detector_comb import SimpleDetectorAggregator
from sklearn.cross_decomposition import PLSRegression
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
            sma_window=3,
            minimal_movement=0.75
        )
    etl.load(dataset)
    etl.preprocess_pooled()
    etl.generate_fourier_dataset(window_overlap=1)

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

    # predefined range of nu for one-class svm
    nu_range = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    for nu in nu_range:
        estimator_list.append(OCSVM(nu=nu, contamination=0.05))

    # predefined range for number of estimators in isolation forests
    n_range = [10, 20, 50, 70, 100, 150, 200, 250]
    for n in n_range:
        estimator_list.append(IForest(n_estimators=n, random_state=42, contamination=0.05))

    return estimator_list

def construct_simple_aggregator(method):
    from combo.models.detector_comb import SimpleDetectorAggregator
    model = SimpleDetectorAggregator(construct_raw_base_estimators(), method=method)
    return model

def construct_xgbod():
    from pyod.models.xgbod import XGBOD
    model = XGBOD(estimator_list=construct_raw_base_estimators(), silent=False, n_jobs=24)
    return model

def construct_lscp():
    from pyod.models.lscp import LSCP
    base_estimators = construct_raw_base_estimators()
    model = LSCP(
        base_estimators,
        local_region_size=150,
        contamination=0.05,
        n_bins=10,
        random_state=42
    )
    return model

def train_model(model_name, X, y, save=True):
    # window_sizes = [128, 256, 512, 1024]
    # angles = ["right_shoulder", "left_shoulder", "right_elbow", "left_elbow", "right_hip", "left_hip", "right_knee",
    #           "left_knee"]
    model_dict = {}
    pls = PLSRegression(n_components=5)
    X = pls.fit_transform(X, y)[0]

    if "lscp" in model_name:
        model = construct_lscp()
    elif "xgbod" in model_name:
        model = construct_xgbod()
    elif "simple-mean" in model_name:
        model = construct_simple_aggregator("average")
    elif "simple-max" in model_name:
        model = construct_simple_aggregator("maximization")

    model.fit(X, y)

    model_dict = {
        "pls": pls,
        "model": model
    }
    if save:
        if not os.path.exists("saved_models"):
            os.mkdir("saved_models")
        save_path = os.path.join("saved_models", model_name + ".joblib")
        joblib.dump(model_dict, save_path)

    return model_dict


if __name__ == "__main__":
    load_data("CIMA")
    train_model("lscp")
