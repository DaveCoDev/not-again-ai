import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns

from not_again_ai.base.file_system import create_file_dir
from not_again_ai.viz.utils import reset_plot_libs


def simple_barplot(
    x: list[str] | (list[float] | (npt.NDArray[np.int_] | npt.NDArray[np.float_])),
    y: list[str] | (list[float] | (npt.NDArray[np.int_] | npt.NDArray[np.float_])),
    save_pathname: str,
    order: str | None = None,
    orient_bars_vertically: bool = True,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    font_size: float = 48,
    height: float = 11,
    aspect: float = 2,
) -> None:
    """Saves a simple barplot to the specified pathname.

    Args:
        x (listlike): Input listlike data for x-axis.
            If orient_bars_vertically is True, this is the category names.
            If orient_bars_vertically is False, this is the bar heights (cannot be numeric).
        y (listlike): Input listlike data for y-axis
            If orient_bars_vertically is True, this is the bar heights (cannot be numeric).
            If orient_bars_vertically is False, this is the category names.
        save_pathname (str): Filepath to save plot to. Parent directories will be automatically created.
        order (str, optional): Order of the bars, either "asc" or "desc". Defaults to None.
        orient_bars_vertically (bool, optional): Whether to orient the bars vertically. Defaults to True.
        title (str, optional): Title of the plot. Defaults to None.
        x_label (str, optional): Label for the x-axis. Defaults to None.
        y_label (str, optional): Label for the y-axis. Defaults to None.
        font_size (float, optional): Font size. Defaults to 48.
        height (float, optional): Height (in inches) of the plot. Defaults to 11.
        aspect (float, optional): Aspect ratio of the plot. Defaults to 2.
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
    sns.set_color_codes("muted")

    if order:
        if order == "asc":
            # sort x and y ascending by y
            if orient_bars_vertically:
                y, x = (list(t) for t in zip(*sorted(zip(y, x, strict=True)), strict=True))
            else:
                x, y = (list(t) for t in zip(*sorted(zip(x, y, strict=True)), strict=True))
        elif order == "desc":
            # sort x and y descending by y
            if orient_bars_vertically:
                y, x = (list(t) for t in zip(*sorted(zip(y, x, strict=True), reverse=True), strict=True))
            else:
                x, y = (list(t) for t in zip(*sorted(zip(x, y, strict=True), reverse=True), strict=True))

    ax = sns.barplot(x=x, y=y, color="b", orient="v" if orient_bars_vertically else "h")
    ax.figure.set_size_inches(height * aspect, height)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    sns.despine()

    create_file_dir(save_pathname)
    plt.savefig(save_pathname, bbox_inches="tight")
    reset_plot_libs()
