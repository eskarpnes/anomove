import pandas as pd
from matplotlib import pyplot as plt

dataframe = pd.read_csv("../../models/results/report/xgbod_groupBy.csv")

dataframe = dataframe[
    (dataframe["window_method"] == "max") &
    (dataframe["angle_method"] == "threshold") &
    (dataframe["body_part_method"] == "mean") &
    (dataframe["window_sizes"] == "[128, 256, 512, 1024]")
]
fig = plt.figure(figsize=[6.4*1.5, 4.8*1.5])
plt.plot(dataframe["threshold"], dataframe["roc_auc"])
plt.title("Threshold scores")
plt.ylabel("roc_auc score")
plt.xlabel("threshold")
plt.xticks([x/10 for x in range(11)])
plt.grid(True)
# plt.savefig("xgbod_threshold.png")

dataframe = pd.read_csv("../../models/results/report/lscp_groupBy.csv")

dataframe = dataframe[
    (dataframe["window_method"] == "max") &
    (dataframe["angle_method"] == "threshold") &
    (dataframe["body_part_method"] == "mean") &
    (dataframe["window_sizes"] == "[128, 256, 512, 1024]")
]

plt.plot(dataframe["threshold"], dataframe["roc_auc"])
#plt.title("Threshold scores - LSCP")
#plt.ylabel("roc_auc score")
#plt.xlabel("threshold")
#plt.xticks([x/10 for x in range(11)])
#plt.grid(True)
#plt.savefig("lscp_threshold.png")

dataframe = pd.read_csv("../../models/results/report/simple-mean_groupBy.csv")

dataframe = dataframe[
    (dataframe["window_method"] == "max") &
    (dataframe["angle_method"] == "threshold") &
    (dataframe["body_part_method"] == "mean") &
    (dataframe["window_sizes"] == "[128, 256, 512, 1024]")
]

plt.plot(dataframe["threshold"], dataframe["roc_auc"])
#plt.title("Threshold scores - Simple mean")
#plt.ylabel("roc_auc score")
#plt.xlabel("threshold")
#plt.xticks([x/10 for x in range(11)])
#plt.grid(True)
#plt.savefig("simple_mean_threshold.png")

dataframe = pd.read_csv("../../models/results/report/simple-max_groupBy.csv")

dataframe = dataframe[
    (dataframe["window_method"] == "max") &
    (dataframe["angle_method"] == "threshold") &
    (dataframe["body_part_method"] == "mean") &
    (dataframe["window_sizes"] == "[128, 256, 512, 1024]")
]

plt.plot(dataframe["threshold"], dataframe["roc_auc"])
#plt.title("Threshold scores - Simple max")
#plt.ylabel("roc_auc score")
#plt.xlabel("threshold")
#plt.xticks([x/10 for x in range(11)])
#plt.grid(True)

plt.legend(["XGBOD", "LSCP", "simple-mean", "simple-max"])

plt.savefig("threshold.png")
