from etl.etl import ETL
import pandas as pd
from sklearn import model_selection, neighbors, metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

parameters = {
    "noise_reduction": ["movement", "short_vector"],
    "pooling": ["mean", "max"],
    "bandwidth": [None, 5, 10],
    "pca": [None, 5, 10],
    "window_size": [128, 256, 512, 1024]
}

results = pd.DataFrame(
    columns=["noise_reduction", "bandwidth", "pooling", "pca", "window_size", "sensitivity", "specificity"]
)


def knn(data, labels, test_size=0.25):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=test_size)

    knn = neighbors.KNeighborsClassifier(n_neighbors=1)

    knn.fit(x_train, y_train)

    pred = knn.predict(x_test)

    tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return sensitivity, specificity


grid = model_selection.ParameterGrid(parameters)

for params in grid:
    print(params)
