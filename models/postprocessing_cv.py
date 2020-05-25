from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import roc_auc_score, roc_curve
from multiprocessing import Pool, Manager
import pandas as pd
import numpy as np
import os
import joblib
from model_training import train_model
from predictor import Predictor
from etl.etl import ETL
from hashlib import sha1
from tqdm import tqdm
from matplotlib import pyplot as plt
DATA_PATH = "/home/erlend/datasets"


def cv(model_name):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    angles = ["right_shoulder", "left_shoulder", "right_elbow", "left_elbow", "right_hip", "left_hip", "right_knee",
              "left_knee"]
    window_sizes = [128, 256, 512, 1024]

    etl = ETL(
        DATA_PATH,
        [128, 256, 512, 1024],
        sma_window=3,
        minimal_movement=0.75
    )

    etl.load("CIMA")

    infants = np.array(list(etl.cima.keys()))
    labels = np.array([etl.cima[infant]["label"] for infant in infants])

    etl.preprocess_pooled()
    etl.generate_fourier_dataset(window_overlap=1)

    X = pd.DataFrame()
    for train_index, test_index in kf.split(infants, labels):

        ids = infants[train_index]

        id_hash = f"{model_name}_{sha1(ids).hexdigest()[:5]}"
        model_path = f"saved_models/{id_hash}.joblib"

        if os.path.exists(model_path):
            models = joblib.load(model_path)
        else:
            models = {}
            for window_size in window_sizes:
                for angle in angles:
                    fourier_path = os.path.join(DATA_PATH, str(window_size), angle + ".json")
                    df = pd.read_json(fourier_path)
                    X = X.append(df)
                X = X[X.id.isin(ids)]
                y = X["label"]
                X = pd.DataFrame(X.data.tolist())

                # model_name = f"{window_size}_{model_name}"
                models[window_size] = train_model(model_name, X, y, save=False)
            joblib.dump(models, model_path)

        x_test = infants[test_index]
        y_test = labels[test_index]

        score = evaluate_model(id_hash, models, x_test, y_test)

def evaluate_model(id_hash, model, X, y):
    print("scoring")
    predictor = Predictor(verbose=True)
    predictor.model = model
    data_path = os.path.join(DATA_PATH, "CIMA", "data")
    # methods = ["mean", "max", "mean_of_max", "max_of_mean", "mean_of_threshold", "max_of_threshold"]

    predictions = []
    for infant, y_true in zip(X, y):
        _, prediction = predictor.predict(data_path, infant)
        prediction.set_id(infant)
        prediction.set_label(y_true)
        predictions.append(prediction)

    if not os.path.exists("results/hashes"):
        os.mkdir("results/hashes")

    joblib.dump(predictions, os.path.join("results/hashes", id_hash + ".joblib"))

    # for window_method in ["mean", "max"]:
    #     for angle_method in ["mean", "max"]:
    #         scores = [prediction.score(window_method, angle_method) for prediciton in predictions]
    #         labels = [prediction.true_label for prediction in predictions]
    #         roc_auc = roc_auc_score(labels, scores)
    #         print("ueh")

    # roc_scores = calculate_roc_auc(id_hash, results, methods)

    # results.to_json(f"results/{id_hash}.json")
    # roc_df = pd.DataFrame.from_dict([roc_scores])
    # roc_df.to_json(f"results/{id_hash}_roc.json")
    # return roc_df

def calculate_roc_auc(id_hash, results, methods):
    roc_auc_results = {}
    try:
        for method in methods:
            y_preds = list(results[method])
            y_true = list(results["label"])
            roc_auc = roc_auc_score(y_true, y_preds)
            print(f"Method: {method} - Score: {roc_auc}")
            roc_auc_results[method] = roc_auc
            plot_roc(id_hash, method, y_true, y_preds)
    except:
        print("fuckit, wanna sleep")
    return roc_auc_results

def plot_roc(id_hash, method, y_true, y_preds):
    fig = plt.figure()
    fpr, tpr, thresholds = roc_curve(y_true, y_preds)
    x = y = np.arange(0, 1.1, 0.1)
    plt.plot(fpr, tpr)
    plt.plot(x, y)
    plt.title(method)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.savefig(f"figures/roc_{id_hash}_{method}.png")


def calculate_scores(model_name):
    window_sizes = {
        128: [True, False],
        256: [True, False],
        512: [True, False],
        1024: [True, False]
    }

    window_size_grid = ParameterGrid(window_sizes)

    window_size_combinations = []

    for combination in window_size_grid:
        window_sizes = [key for key, val in combination.items() if val]
        if window_sizes:
            window_size_combinations.append(window_sizes)

    agg_methods = ["mean", "max"]

    folds = []

    print("Loading data from folds")
    for filename in os.listdir("results/hashes"):
        if model_name in filename:
            folds.append(joblib.load(os.path.join("results/hashes", filename)))
    print("Finished loading data")

    results = pd.DataFrame(
        columns=["window_method", "angle_method", "body_part_method", "threshold", "roc_auc"]
    )

    print("Calculating all scores")

    total_steps = len(folds) * len(agg_methods) ** 3 * len(window_size_combinations)
    total_threshold_steps = len(folds) * len(agg_methods) ** 2 * len(range(1, 100)) * len(window_size_combinations)
    total_steps += total_threshold_steps

    pbar = tqdm(total=total_steps)

    for fold in folds:
        for window_method in agg_methods:
            for window_sizes in window_size_combinations:
                for angle_method in agg_methods:
                    for body_part_method in agg_methods:
                        y_pred = []
                        y_true = []
                        for prediction in fold:
                            prediction.set_window_sizes(window_sizes)
                            scores = prediction.score(window_method, angle_method)
                            if body_part_method == "mean":
                                score = np.nanmean(scores)
                            elif body_part_method == "max":
                                score = np.nanmax(scores)
                            y_pred.append(score)
                            y_true.append(prediction.true_label)
                        roc_score = roc_auc_score(y_true, y_pred)
                        results = results.append({
                            "window_method": window_method,
                            "window_sizes": window_sizes,
                            "angle_method": angle_method,
                            "body_part_method": body_part_method,
                            "threshold": "NA",
                            "roc_auc": roc_score
                        }, ignore_index=True)
                        pbar.update()
                        print(f"\nwindow: {window_method}\n"
                              f"window_sizes: {window_sizes}\n"
                              f"angle: {angle_method}\n"
                              f"body_part: {body_part_method}\n"
                              f"score: {roc_score}\n\n")
                for body_part_method in agg_methods:
                    for threshold in range(1, 100):
                        threshold /= 100
                        y_pred = []
                        y_true = []
                        for prediction in fold:
                            prediction.set_window_sizes(window_sizes)
                            scores = prediction.score_threshold(window_method, body_part_method, threshold)
                            if body_part_method == "mean":
                                score = np.nanmean(scores)
                            elif body_part_method == "max":
                                score = np.nanmax(scores)
                            y_pred.append(score)
                            y_true.append(prediction.true_label)
                        roc_score = roc_auc_score(y_true, y_pred)
                        pbar.update()
                        results = results.append({
                            "window_method": window_method,
                            "window_sizes": window_sizes,
                            "angle_method": "threshold",
                            "body_part_method": body_part_method,
                            "threshold": threshold,
                            "roc_auc": roc_score
                        }, ignore_index=True)
                        print(f"\nwindow: {window_method}\n"
                              f"window_sizes: {window_sizes}\n"
                              f"body_part: {body_part_method}\n"
                              f"threshold: {threshold}\n"
                              f"score: {roc_score}\n\n")
        results.to_csv(os.path.join("results/report", model_name + ".csv"))

    return results

if __name__ == "__main__":
    # cv("xgbod")
    # calculate_scores("xgbod")
    cv("lscp")
    calculate_scores("lscp")
    cv("simple-mean")
    calculate_scores("simple-mean")
    cv("simple-max")
    calculate_scores("simple-max")
























    # result_paths = [
    #     "2cbb36249b411eaa9170bf555aa0f0817201338a.json",
    #     "6aa9461a42098990d908edc9fd4d468693b4c20c.json",
    #     "185f35047c100e21dd01df22dad7cb7812af51bc.json",
    #     "e2d42f634e23fb0beb78e93c7c06749911370d5c.json",
    #     "fa6a058fac19ca7897e356afff8c166208a717be.json"
    # ]
    # methods = ["mean", "max", "mean_of_max", "max_of_mean", "mean_of_threshold", "max_of_threshold"]
    # results = pd.DataFrame()
    # for path in result_paths:
    #     results = results.append(pd.read_json(f"results/{path}"))
    # calculate_roc_auc("all_data", results, methods)
