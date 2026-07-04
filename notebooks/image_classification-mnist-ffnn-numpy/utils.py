import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

class Utils:
    mini_batchify = staticmethod(
        lambda sequence, batch_size: (
            sequence[batch_lo : batch_lo + batch_size] for batch_lo in range(0, len(sequence), batch_size)
        )
    )

    one_hot_decode = staticmethod(lambda X: np.argmax(X, axis=1))

    @staticmethod
    def one_hot_encode(X, C):
        class_to_idx = {c: idx for idx, c in enumerate(C)}
        labels = np.asarray(X)
        indices = np.array([class_to_idx[x] for x in labels.reshape(-1)])
        indices = indices.reshape(labels.shape)
        return np.eye(len(C), dtype=int)[indices]

    def two_line_plot(
        fig_size
        , title
        , x_axis_label
        , x
        , y1_legend
        , y_axis_label
        , y1
        , y2_legend
        , y2
        , x_ticks_inc=20
    ):
        plt.figure(figsize = fig_size)

        ax = sns.lineplot(label=y1_legend, x=x, y=y1)
        ax = sns.lineplot(label=y2_legend, x=x, y=y2)

        ax.set_xlim(0, len(x))
        ax.set_title(title, fontsize=14)
        ax.set_ylabel(y_axis_label, fontsize=14)
        ax.set_xlabel(x_axis_label, fontsize=14)
        ax.set_xticks(range(0, len(x), x_ticks_inc))

        plt.legend(fontsize='x-large')
        plt.tight_layout()
        plt.show()
