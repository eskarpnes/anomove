from etl.etl import ETL
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

window_sizes = [128, 256, 512, 1024]
etl = ETL("/home/erlend/datasets", window_sizes)
etl.load("CIMA")
etl.preprocess_pooled()
angles = etl.angles.keys()
differences = {}


for window_size in tqdm(window_sizes):
    etl.differences = []
    for angle in angles:
        etl.generate_fourier_data(angle, window_size, window_size)
    differences[window_size] = etl.differences

bins = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]

for window_size in window_sizes:
    fig = plt.figure()
    plt.hist(
        differences[window_size],
        color="mediumslateblue",
        bins=bins,
        weights=np.ones(len(differences[window_size])) / len(differences[window_size]),
        edgecolor="black",
        linewidth=0.5
    )
    plt.xticks(bins)
    plt.xlim(0, 2.5)
    plt.ylim(0, 0.35)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel(f"Movement in window (radians)")
    plt.ylabel(f"Percentage of windows")
    plt.title(f"Window size: {str(window_size)}")
    plt.savefig(f"{str(window_size)}-movement.png")

