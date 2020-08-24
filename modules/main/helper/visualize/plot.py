import matplotlib.pyplot as plt
from os.path import join, exists
from os import makedirs
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(
    join(os.path.abspath(__file__), '..', '..', '..', '..')))


def plot_history(history):
    '''
        Plotting the
        :arguments history: data to plot
        :return: None
    '''
    from config import cfg
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)

    figure_path = join(cfg.vis_dir, 'training_plot.png')
    if not exists(cfg.vis_dir):
        makedirs(cfg.vis_dir)

    plt.savefig(figure_path)

    plt.show()


if __name__ == "__main__":
    for path in sys.path:
        print(path)
