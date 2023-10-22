import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns

from not_again_ai.base.file_system import create_file_dir
from not_again_ai.viz.utils import reset_plot_libs


def scatterplot_basic(
    x: list[float] | (npt.NDArray[np.int_] | npt.NDArray[np.float_]),
    y: list[float] | (npt.NDArray[np.int_] | npt.NDArray[np.float_]),
    save_pathname: str,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    font_size: float = 48,
    height: float = 13,
    aspect: float = 1.2,
) -> None:
    """Saves a basic scatterplot to the specified pathname.

    Args:
        x (listlike): Input listlike data for x-axis
        y (listlike): Input listlike data for y-axis
        save_pathname (str): Filepath to save plot to. Parent directories will be automatically created.
        title (str, optional): Title of the plot. Defaults to None.
        xlim (tuple[float, float], optional): Set the x-axis limits (lower, upper). Defaults to None.
        ylim (tuple[float, float], optional): Set the y-axis limits (lower, upper). Defaults to None.
        font_size (float, optional): Font size. Defaults to 48.
        height (float, optional): Height (in inches) of the plot. Defaults to 13.
        aspect (float, optional): Aspect ratio of the plot, so that `aspect` * `height` gives the width of each facet in inches. Defaults to 1.2.
    """

    sns.set_theme(
        style="white",
        rc={
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size * 0.8,
            "xtick.labelsize": font_size * 0.65,
            "ytick.labelsize": font_size * 0.65,
            "legend.fontsize": font_size * 0.5,
            "legend.title_fontsize": font_size * 0.55,
        },
    )
    data = pd.DataFrame({"x": x, "y": y})
    ax = sns.scatterplot(data=data, x="x", y="y")

    ax.figure.set_size_inches(height * aspect, height)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    sns.despine(top=True, right=True)

    create_file_dir(save_pathname)
    plt.savefig(save_pathname, bbox_inches="tight")
    reset_plot_libs()
