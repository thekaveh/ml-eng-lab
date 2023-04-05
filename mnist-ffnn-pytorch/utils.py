import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

class Utils:
    @staticmethod
    def multi_line_plot(fig_size, title, x_axis_label, x, y_axis_label, yss_legend, yss, x_ticks_inc=20):
        _ = plt.figure(figsize = fig_size)
        
        ls = ["-", "--", "-.", ":"]
        cs = sns.color_palette(n_colors=len(yss))

        for ys_idx, (ys, ys_legend) in enumerate(zip(yss, yss_legend)):
            for y_idx, y in enumerate(ys):
                ax = sns.lineplot(
                    x=x
                    , y=y
                    , color=cs[ys_idx]
                    , linestyle=ls[y_idx]
                    , label=ys_legend[y_idx]
                )

        ax.set_xlim(0, len(x))
        ax.set_title(title, fontsize=14)
        ax.set_ylabel(y_axis_label, fontsize=14)
        ax.set_xlabel(x_axis_label, fontsize=14)
        ax.set_xticks(range(0, len(x), x_ticks_inc))
        
        plt.legend(fontsize='large')
        plt.tight_layout()
        plt.show();

    @staticmethod
    def get_device_name():
        return (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "gpu" if torch.cuda.is_available()
            else "cpu"
        )