import matplotlib.pyplot as plt
import numpy as np
import matplotlib

x1 = np.linspace(0.0, 0.5)
data1 = np.cos(2 * np.pi * x1)

x2 = np.linspace(0.0, 0.5)
data2 = np.cos(4 * np.pi * x1)

x3 = np.linspace(0.0, 0.5)
data3 = np.cos(8 * np.pi * x1)


class graph:

    def __init__(self, data, title, x_label, y_label, figsize=[20,9], save=False, grid=False):
        self.data = data
        self.title = title
        self.figsize = figsize
        self.y_label = y_label
        self.x_label = x_label
        self.save = save
        self.grid = grid
        self.colors = ["red", "green", "blue"]


def plot_graph(title, y_label, x_label, data, figsize=[20, 9], subplots=None, save=False, grid=False):
    colors = ["red", "green", "blue", "orange", "yellow"]
    if subplots is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(data)
        set_properties(ax, title, x_label, y_label, grid)
    else:
        rows = subplots[0]
        columns = subplots[1]
        fig, ax = plt.subplots(rows, columns, figsize=figsize)
        axes = ax.flatten()
        for i in range(len(axes)):
            axes[i].plot(data[i], color=colors[i])
            set_properties(axes[i], title, x_label, y_label, grid)
    if save is True:
        fig.savefig(title + ".png")

    plt.show()


def plot_scatter(title, x_label, y_label, x_data, y_data, figsize=[20, 9], subplots=None, save=False, grid=False):
    colors = ["red", "green", "blue", "orange", "yellow"]
    if subplots is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(x_data, y_data, color=colors[0])
        set_properties(ax, title, x_label, y_label, grid)
    else:
        rows = subplots[0]
        columns = subplots[1]
        fig, ax = plt.subplots(rows, columns, figsize=figsize)
        axes = ax.flatten()
        for i in range(len(axes)):
            axes[i].scatter(x_data[i], y_data[i], color=colors[i])
            set_properties(axes[i], title, x_label, y_label, grid)
    if save:
        fig.savefig(title + ".png")

    plt.show()


def set_properties(ax, title, x_label, y_label, grid):
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(grid)

data_agg = [data1, data2, data3]

data_x = [data1, data2]
data_y = [data2, data3]

plot_scatter("testing", "y", "x", data1, data2,)