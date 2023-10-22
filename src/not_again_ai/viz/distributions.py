import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns

from not_again_ai.base.file_system import create_file_dir
from not_again_ai.viz.utils import reset_plot_libs


def univariate_distplot(
    data: list[float] | npt.NDArray[np.float64],
    save_pathname: str,
    print_summary: bool = True,
    title: str | None = None,
    xlabel: str | None = "Value",
    ylabel: str | None = "Count",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: npt.ArrayLike | None = None,
    yticks: npt.ArrayLike | None = None,
    bins: int = 50,
    font_size: float = 48,
    height: float = 13,
    aspect: float = 2.2,
) -> None:
    """Saves a univariate distribution plot to the specified pathname.

    Args:
        data (listlike): Input listlike data to plot distribution of
        save_pathname (str): Filepath to save plot to. Parent directories will be automatically created.
        print_summary (bool, optional): If true will print summary statistics. Defaults to True.
        title (str, optional): Title of the plot. Defaults to None.
        xlabel (str, optional): Set the label for the x-axis. Defaults to 'Value'.
        ylabel (str, optional): Set the label for the y-axis. Defaults to 'Count'.
        xlim (tuple[float, float], optional): Set the x-axis limits (lower, upper). Defaults to None.
        ylim (tuple[float, float], optional): Set the y-axis limits (lower, upper). Defaults to None.
        xticks (npt.ArrayLike, optional): Set the x-axis tick locations. Defaults to None.
        yticks (npt.ArrayLike, optional): Set the y-axis tick locations. Defaults to None.
        bins (int, optional): See matplotlib [histplot documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html#matplotlib-pyplot-hist) for all options. Defaults to 50.
        font_size (float, optional): Font size. Defaults to 48.
        height (float, optional): Height (in inches) of the plot. Defaults to 13.
        aspect (float, optional): Aspect ratio of the plot, so that `aspect` * `height` gives the width of each facet in inches. Defaults to 2.2.
    """

    sns.set_theme(
        style="white",
        rc={
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size * 0.8,
            "xtick.labelsize": font_size * 0.7,
            "ytick.labelsize": font_size * 0.7,
            "legend.fontsize": font_size * 0.55,
        },
    )

    # precompute summary statistics
    mean = np.mean(data)
    median = np.median(data)
    stdev = np.std(data)
    percentile_5 = np.percentile(data, 5)
    percentile_95 = np.percentile(data, 95)

    facet_grid = sns.displot(data, bins=bins, height=height, aspect=aspect)

    facet_grid.set(
        xlim=xlim,
        ylim=ylim,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )

    if xticks is not None:
        facet_grid.set(xticks=xticks)
    if yticks is not None:
        facet_grid.set(yticks=yticks)

    pastel_colors = sns.color_palette("pastel")

    ax = facet_grid.axes.flatten()[0]

    # plot summary statistic lines
    ax.axvline(x=mean, color=pastel_colors[1], ls="--", lw=2.5, label=f"Mean: {mean:.3f}")
    ax.axvline(x=median, color=pastel_colors[2], ls="--", lw=2.5, label=f"Median: {mean:.3f}")
    ax.axvline(x=percentile_5, color=pastel_colors[9], ls="--", lw=2.5, label=f"5 Percentile: {percentile_5:.3f}")
    ax.axvline(x=percentile_95, color=pastel_colors[4], ls="--", lw=2.5, label=f"95 Percentile: {percentile_95:.3f}")
    # add legend for these lines
    handles, _ = ax.get_legend_handles_labels()
    # and an empty patch for the stdev statistic
    handles.append(mpatches.Patch(color="none", label=f"St Dev: {stdev:.3f}"))

    plt.legend(handles=handles, loc=0)

    create_file_dir(save_pathname)
    plt.savefig(save_pathname, bbox_inches="tight")
    reset_plot_libs()

    if print_summary:
        to_print = (
            "Summary Statistics:",
            f"Mean:\t\t{mean:.3f}",
            f"Median:\t\t{median:.3f}",
            f"5 Percentile:\t{percentile_5:.3f}",
            f"95 Percentile:\t{percentile_95:.3f}",
            f"St Dev:\t\t{stdev:.3f}",
        )
        print("\n".join(to_print))
