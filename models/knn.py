import numpy as np
import pandas as pd
import os
from sklearn import model_selection, neighbors, metrics
import matplotlib.pyplot as plt
from tqdm import tqdm


def graph_knn(angle, window_size):
    DATA_PATH = "/home/erlend/datasets/" + window_size + "/" + angle + ".json"

    df = pd.read_json(DATA_PATH)

    data = pd.DataFrame(list(df["data"]))

    X_train, X_test, y_train, y_test = model_selection.train_test_split(data, df["label"], test_size=0.25)

    # knn = neighbors.KNeighborsClassifier(n_neighbors=10)

    # knn.fit(X_train, y_train)

    # pred = knn.predict(X_test)

    # print(metrics.confusion_matrix(y_test, pred))
    # print(metrics.classification_report(y_test, pred))

    error_rate = []
    true_positives = []

    # Will take some time
    for i in tqdm(range(1, 40)):
        knn = neighbors.KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        true_pos = metrics.confusion_matrix(y_test, pred_i)[1][1]
        true_positives.append(true_pos)
        error_rate.append(np.mean(pred_i != y_test) * 100)

    if not os.path.exists(window_size):
        os.mkdir(window_size)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=5)
    plt.title('Error rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.savefig(window_size + "/error_rate_" + angle + ".png")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 40), true_positives, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=5)
    plt.title('True positive vs. K Value')
    plt.xlabel('K')
    plt.ylabel('True positives')
    plt.savefig(window_size + "/true_positives_" + angle + ".png")


angles = ["V1", "V2", "V3", "V4", "V5", "V6"]
window_sizes = ["128", "256", "512", "1024"]

for window_size in window_sizes:
    for angle in angles:
        graph_knn(angle, window_size)
