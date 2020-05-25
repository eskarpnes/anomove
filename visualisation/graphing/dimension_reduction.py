import pandas as pd
import os
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA


DATA_PATH = "/home/erlend/datasets"
angles = ["right_shoulder", "left_shoulder", "right_elbow", "left_elbow", "right_hip", "left_hip", "right_knee",
              "left_knee"]
window_sizes = [128, 256, 512, 1024]

for window_size in window_sizes:
    X = pd.DataFrame()
    for angle in angles:
        fourier_path = os.path.join(DATA_PATH, str(window_size), angle + ".json")
        df = pd.read_json(fourier_path)
        X = X.append(df)
    y = X["label"]
    X = pd.DataFrame(X.data.tolist())
    pls = PLSRegression(n_components=10)
    pls.fit(X, y)
    pca = PCA(n_components=10)
    pca.fit(X)
    print(f"PCA variance ratio: {pca.explained_variance_ratio_}")
