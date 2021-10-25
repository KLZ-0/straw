import sys
from os import path
from pathlib import Path

import numpy as np

# only for debug
import matplotlib.pyplot as plt

FIG_SIZE = (10, 2.5)


def plot_list(datalist, filename, figsize=FIG_SIZE, title="Title", xlabel="x", ylabel="y", plot_labels=None,
              xspan=(0., -1.)):
    """
    Function ported from my ISS project: https://github.com/KLZ-0/ISS-project/
    :param datalist: data in the format [signal1, signal2, ..., signaln]
    :param filename: save file name including suffix
    :param figsize: figure size (width, height)
    :param title: figure title
    :param xlabel: X axis label
    :param ylabel: Y axis label
    :param plot_labels: Labels for the given data in order ["Signal 1", "Signal 2", ..., "Signal n"]
    :param xspan: X axis numbering (left/min, right/max)
    :return: None
    """
    if plot_labels and len(plot_labels) != len(datalist):
        print("Data list size does not match label list size, wtf?", file=sys.stderr)
        return

    if float(xspan[1]) != -1.:
        time = np.linspace(float(xspan[0]), float(xspan[1]), datalist[0].shape[0])
    else:
        time = np.linspace(float(xspan[0]), datalist[0].shape[0] - 1, datalist[0].shape[0])

    plt.figure(figsize=figsize)
    for i in range(len(datalist)):
        if plot_labels:
            plt.plot(time, datalist[i], label=plot_labels[i])
        else:
            plt.plot(time, datalist[i])

    plt.gca().set_xlabel(xlabel)
    plt.gca().set_ylabel(ylabel)

    plt.gca().set_title(title)
    plt.tight_layout()

    if plot_labels:
        plt.legend()

    Path("outputs").mkdir(parents=True, exist_ok=True)
    plt.savefig(path.join("outputs", filename))
