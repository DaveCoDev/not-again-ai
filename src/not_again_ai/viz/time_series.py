import matplotlib
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns

from not_again_ai.base.file_system import create_file_dir
from not_again_ai.viz.utils import reset_plot_libs


def ts_lineplot(
    ts_data: list[float] | (npt.NDArray[np.float64] | npt.NDArray[np.int64]),
    save_pathname: str,
    ts_x: (
        list[float]
        | (npt.NDArray[np.float64] | (npt.NDArray[np.datetime64] | (npt.NDArray[np.int64] | pd.Series)))
        | None
    ) = None,
    ts_names: list[str] | None = None,
    title: str | None = None,
    xlabel: str | None = "Time",
    ylabel: str | None = "Value",
    legend_title: str | None = None,
    xaxis_date_format: str | None = None,
    xaxis_major_locator: matplotlib.ticker.Locator | None = None,
    ylim: tuple[float, float] | None = None,
    yticks: npt.ArrayLike | None = None,
    font_size: float = 48,
    height: float = 13,
    aspect: float = 2.2,
    linewidth: float = 2,
    legend_loc: str | (tuple[float, float] | int) | None = None,
    palette: str | (list[str] | (list[float] | (dict[str, str] | matplotlib.colors.Colormap))) = "tab10",
) -> None:
    """Saves a time series plot where each row in `ts_data` is a time series.
    Optionally, a specific x axis (like dates) can be provided with `ts_x`.
    Names to appear in the legend for each time series can be provided with `ts_names`.

    Args:
        ts_data (list of lists or 2D numpy array): Each nested list or row is a time series to be plotted.
        save_pathname (str): Filepath to save plot to. Parent directories will be automatically created.
        ts_x (listlike, optional): The values that will be used for the x-axis. Defaults to None.
        ts_names (list[str], optional): The names of the time series shown on the legend. Defaults to None.
        title (str, optional): Title of the plot. Defaults to None.
        xlabel (str, optional): Set the label for the x-axis. Defaults to 'Time'.
        ylabel (str, optional): Set the label for the y-axis. Defaults to 'Value'.
        legend_title (str, optional): Sets the title of the legend. Defaults to None.
        xaxis_date_format (str, optional): A dateformat string. See [strftime cheatsheet](https://strftime.org/). Defaults to None.
        xaxis_major_locator (matplotlib.ticker.Locator, optional): Matplotlib tick locator,
            See [Tick locating](https://matplotlib.org/stable/api/ticker_api.html) or [Date tickers](https://matplotlib.org/stable/api/ticker_api.html) for the available options. Defaults to None.
        ylim (tuple[float, float], optional): Set the y-axis limits (lower, upper). Defaults to None.
        yticks (npt.ArrayLike, optional): Set the y-axis tick locations. Defaults to None.
        font_size (float, optional): Font size. Defaults to 48.
        height (float, optional): Height (in inches) of the plot. Defaults to 13.
        aspect (float, optional): Aspect ratio of the plot, so that `aspect` * `height` gives the width of each facet in inches. Defaults to 2.2.
        linewidth (float, optional): Size of each time series line. Defaults to 2.
        legend_loc (Union[str, tuple[float, float], int], optional): Matplotlib legend location.
            See [matplotlib documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html). Defaults to None.
        palette (str, list, dict, matplotlib.colors.Colormap], optional): Takes the same arguments as [seaborn's lineplot](https://seaborn.pydata.org/generated/seaborn.lineplot.html#seaborn-lineplot) palette argument.
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
    # Transpose the list of lists or numpy array
    ts_data = np.array(ts_data).T
    sns_data = pd.DataFrame(ts_data, columns=ts_names)
    if ts_x is None:
        ts_x = np.arange(len(ts_data))

    sns_data["Time"] = ts_x
    sns_data = sns_data.melt(id_vars="Time", var_name="Time Series", value_name="Value")
    ax = sns.lineplot(data=sns_data, x="Time", y="Value", hue="Time Series", palette=palette, linewidth=linewidth)

    ax.figure.set_size_inches(height * aspect, height)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title=legend_title)

    ax.set_ylim(ylim)

    if legend_loc is not None:
        ax.legend(loc=legend_loc)

    if (xaxis_date_format is not None) and (ts_x is not None):
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(xaxis_date_format))

    if (xaxis_major_locator is not None) and (ts_x is not None):
        ax.xaxis.set_major_locator(xaxis_major_locator)

    if yticks is not None:
        ax.set(yticks=yticks)

    sns.despine(top=True, right=True)

    create_file_dir(save_pathname)
    plt.savefig(save_pathname, bbox_inches="tight")
    reset_plot_libs()
