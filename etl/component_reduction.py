from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "/home/erlend/datasets/256/"

angles = ["V3", "V4", "V5", "V6"]

fig, axes = plt.subplots(2, 2, figsize=(20, 20), sharex=True, sharey=True)

targets = [0, 1]
colors = ["g", "r"]

for i in range(len(angles)):
    df = pd.read_json(DATA_PATH + angles[i] + ".json")
    df_features = pd.DataFrame(df.data.tolist())
    df_features = StandardScaler().fit_transform(df_features)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_features)
    principal_df = pd.DataFrame(data=principal_components, columns=["component 1", "component 2"])
    principal_df = pd.concat([principal_df, df[["label"]]], axis=1)

    # axes[i // 2][i % 2].plot(np.cumsum(pca.explained_variance_ratio_))

    for target, color in zip(targets, colors):
        indices = principal_df['label'] == target
        axes[i // 2][i % 2].scatter(principal_df.loc[indices, 'component 1'],
                                    principal_df.loc[indices, 'component 2'],
                                    c=color,
                                    alpha=0.25,
                                    s=50)
    axes[i // 2][i % 2].title.set_text(angles[i])
    axes[i // 2][i % 2].legend(targets)
    axes[i // 2][i % 2].grid()
plt.show()
