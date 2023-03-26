import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

class Utils:
    def two_line_plot(fig_size, title, x_axis_label, x, y1_legend, y_axis_label, y1, y2_legend, y2):
        fig = plt.figure(figsize = fig_size)

        ax = sns.lineplot(label=y1_legend, x=x, y=y1)
        ax = sns.lineplot(label=y2_legend, x=x, y=y2)

        ax.set_xlim(0, len(x))
        ax.set_title(title, fontsize=14)
        ax.set_ylabel(y_axis_label, fontsize=14)
        ax.set_xlabel(x_axis_label, fontsize=14)
        ax.set_xticks(range(0, len(x), 10))

        plt.legend(fontsize='x-large')
        plt.tight_layout()
        plt.show();

    def get_device_name():
        return (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "gpu" if torch.cuda.is_available()
            else "cpu"
        )