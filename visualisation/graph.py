import matplotlib.pyplot as plt
import os


def get_colors():
    return ["red",
            "green",
            "blue",
            "orange",
            "yellow",
            "cyan",
            "magenta",
            "black",
            "lawngreen",
            "navy"
            ]


def set_properties(ax, title, x_label, y_label, x_lim, y_lim, grid):
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.grid(grid)


def plot_graph(title, x_label, y_label, data, figsize=[20, 9], x_lim=None, y_lim=None, subplots=None, save=False,
               grid=False, show=True, save_path=""):
    colors = get_colors()
    if subplots is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(data, color=colors[0])
        set_properties(ax, title, x_label, y_label, x_lim, y_lim, grid)
    else:
        rows = subplots[0]
        columns = subplots[1]
        fig, ax = plt.subplots(rows, columns, figsize=figsize)
        axes = ax.flatten()
        for i in range(len(axes)):
            axes[i].plot(data[i], color=colors[i])
            set_properties(axes[i], title, x_label, y_label, x_lim, y_lim, grid)
    if save:
        path = os.path.join(save_path, title + ".png")
        fig.savefig(path)
        plt.close()

    if show:
        plt.show()


def plot_scatter(title, x_label, y_label, x_data, y_data, figsize=[20, 9], x_lim=None, y_lim=None, alpha=1, size=10,
                 subplots=None, save=False, grid=False, show=True, save_path=""):
    colors = get_colors()
    if subplots is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(x_data, y_data, color=colors[0])
        set_properties(ax, title, x_label, y_label, x_lim, y_lim, grid)
    else:
        rows = subplots[0]
        columns = subplots[1]
        fig, ax = plt.subplots(rows, columns, figsize=figsize)
        axes = ax.flatten()
        for i in range(len(axes)):
            axes[i].scatter(x_data[i], y_data[i], color=colors[i], alpha=alpha, s=size)
            set_properties(axes[i], title, x_label, y_label, x_lim, y_lim, grid)
    if save:
        path = os.path.join(save_path, title + ".png")
        fig.savefig(path)
        plt.close()

    if show:
        plt.show()
