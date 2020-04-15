from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "/home/erlend/datasets/128/"

angles = ["shoulder", "elbow", "hip", "knee"]

height = 4
width = 10
plots = height*width

# fig, axes = plt.subplots(height, width, figsize=(20, 20), sharex=True, sharey=True)

colors = {
    0: "g",
    1: "r"
}

df = pd.read_json(DATA_PATH + "left_shoulder.json")
df_features = pd.DataFrame(df.data.tolist())

plsr = PLSRegression(n_components=2)
X = df_features
y = df["label"]
principal_components = plsr.fit_transform(X, y)
principal_df = pd.DataFrame(data=principal_components[0], columns=["component 1", "component 2"])
principal_df = pd.concat([principal_df, df[["label"]]], axis=1)
principal_df = pd.concat([principal_df, df[["id"]]], axis=1)

# axes[i // 2][i % 2].plot(np.cumsum(pca.explained_variance_ratio_))

healthy_id = list(set(principal_df.loc[df["label"] == 0]["id"]))
impaired_id = list(set(principal_df.loc[df["label"] == 1]["id"]))

healthy_ids = np.random.choice(healthy_id, plots//2)
impaired_ids = np.random.choice(impaired_id, plots//2)

all_ids = np.append(healthy_ids, impaired_ids)

# for i in range(len(all_ids)):
#     infant_id = all_ids[i]
#     indices = principal_df['id'] == infant_id
#     target = list(principal_df.loc[principal_df["id"] == infant_id]["label"])[0]
#     axes[i // width][i % width].scatter(
#         principal_df.loc[indices, 'component 1'],
#         principal_df.loc[indices, 'component 2'],
#         c=colors[target],
#         alpha=0.25,
#         s=50
#     )
#     axes[i // width][i % width].title.set_text(infant_id)
#     axes[i // width][i % width].legend(infant_id)
#     axes[i // width][i % width].grid()

fig = plt.figure(figsize=[20, 20])

for ids in [healthy_ids, impaired_ids]:
    ids = list(ids)
    indices = principal_df["id"].isin(ids)
    target = list(principal_df.loc[principal_df["id"].isin(ids)]["label"])[0]
    plt.scatter(
        principal_df.loc[indices, "component 1"],
        principal_df.loc[indices, "component 2"],
        c=colors[target],
        alpha=0.25,
        s=50
    )

plt.show()
