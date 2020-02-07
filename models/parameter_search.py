import sys
import time
sys.path.append('../')
from etl.etl import ETL
import pandas as pd
from sklearn import model_selection, neighbors, metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_PATH = "/home/erlend/datasets/"

parameters = {
    "noise_reduction": ["movement", "short_vector", "all"],
    "pooling": ["mean", "max"],
    "bandwidth": [None, 5, 10],
    "pca": [None, 5, 10],
    "k": [1, 3]
}

results = pd.DataFrame(
    columns=["noise_reduction", "bandwidth", "pooling", "pca", "k", "window_size", "angle", "sensitivity", "specificity"]
)


def knn(data, labels, k=1, test_size=0.25):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=test_size)

    knn = neighbors.KNeighborsClassifier(n_neighbors=k)

    knn.fit(x_train, y_train)

    pred = knn.predict(x_test)

    tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    if tp+fn == 0:
        sensitivity = 0
    else:
        sensitivity = tp / (tp + fn)

    if tn+fp == 0:
        specificity = 0
    else:
        specificity = tn / (tn + fp)

    return sensitivity, specificity



grid = model_selection.ParameterGrid(parameters)
window_sizes = [128, 256, 512, 1024]


print(f"{len(grid)} different combinations of parameters will be explored.")

for i, params in enumerate(grid):
    etl = ETL(
        data_path=DATA_PATH,
        window_sizes=window_sizes,
        bandwidth=params["bandwidth"],
        pooling=params["pooling"],
        noise_reduction=params["noise_reduction"]
    )
    etl.load("CIMA_angles_resampled_cleaned", tiny=False)
    etl.generate_fourier_dataset()

    angles = ["V1", "V2", "V3", "V4", "V5", "V6"]
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

                sensitivity, specificity = knn(data, labels, k=params["k"])
            except ValueError:
                sensitivity, specificity = ("crashed", "crashed")

            results = results.append({
                "noise_reduction": params["noise_reduction"],
                "bandwidth": params["bandwidth"],
                "pooling": params["pooling"],
                "pca": params["pca"],
                "k": params["k"],
                "window_size": str(window_size),
                "angle": angle,
                "sensitivity": sensitivity,
                "specificity": specificity
            }, ignore_index=True)
    # Short rest to prevent race conditions.
    time.sleep(1)
    print(f"{i+1} of {len(grid)} runs done.")

results.to_csv("results.csv")
