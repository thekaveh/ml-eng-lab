import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

class Utils:
    @staticmethod
    def multi_line_plot(fig_size, title, x_axis_label, x, y_axis_label, yss_legend, yss, x_ticks_inc=20, title_size=15, label_size=12):
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
        ax.set_title(title, fontsize=title_size)
        ax.set_ylabel(y_axis_label, fontsize=label_size)
        ax.set_xlabel(x_axis_label, fontsize=label_size)
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
        
    @staticmethod
    def scatter_plot(vm, figsize=(15, 15), title_size=15, label_size=12):
        f, sub_plt = plt.subplots(nrows=1, ncols=1, figsize=figsize)

        for t_idx, _ in enumerate(vm["ts"]["uni_vals"]):
            sub_plt.scatter(
                s=10
                , x=vm["xs-ts"][t_idx]
                , y=vm["ys-ts"][t_idx]
                , color=vm["ts"]["colors"][t_idx]
                , label=vm["ts"]["labels"][t_idx]
            )

        sub_plt.set_xticks([])
        sub_plt.set_yticks([])
        sub_plt.set_xlabel(vm["xs"]["label"], fontsize=label_size)
        sub_plt.set_ylabel(vm["ys"]["label"], fontsize=label_size)
        sub_plt.set_title(vm["title"], fontsize=title_size)
        sub_plt.legend(fontsize='large')
        plt.tight_layout();

    @staticmethod
    def get_scatter_plot_vm(data, title, col_xs, label_xs, col_ys, label_ys, col_ts, labels_ts, colors_ts, uni_ts):
        vm = {
            "title": title,
            "xs": {
                "vals": data[col_xs],
                "label": label_xs
            },
            "ys": {
                "vals": data[col_ys],
                "label": label_ys
            },
            "ts": {
                "vals": data[col_ts],
                "uni_vals": uni_ts,
                "colors": colors_ts,
                "labels": labels_ts
            }
        }

        vm["xs-ts"] = [vm["xs"]["vals"][vm["ts"]["vals"] == t_val] for t_val in vm["ts"]["uni_vals"]]
        vm["ys-ts"] = [vm["ys"]["vals"][vm["ts"]["vals"] == t_val] for t_val in vm["ts"]["uni_vals"]]

        return vm