import matplotlib
import numpy as np
import pandas as pd

from not_again_ai.viz.time_series import ts_lineplot


def test_ts_lineplot() -> None:
    rs = np.random.RandomState(365)
    values = rs.randn(365, 4).cumsum(axis=0).T
    dates = pd.date_range("1 1 2021", periods=365, freq="D")
    ts_lineplot(
        ts_data=values,
        save_pathname=".nox/temp/ts_lineplot1.png",
        ts_x=None,
        ts_names=None,
        legend_title="Test Title",
    )
    ts_lineplot(
        ts_data=values,
        save_pathname=".nox/temp/ts_lineplot2.png",
        ts_x=dates,
        ts_names=None,
        xlabel="Months",
        ylabel="Number",
        ylim=(-15, 15),
    )
    ts_lineplot(
        ts_data=values,
        save_pathname=".nox/temp/ts_lineplot3.png",
        ts_x=None,
        ts_names=["A", "B", "C", "D"],
        height=14,
        aspect=1.5,
    )
    ts_lineplot(
        ts_data=values,
        save_pathname=".nox/temp/ts_lineplot4.png",
        ts_x=dates,
        ts_names=["A", "B", "C", "D"],
        title="Example Time Series",
        xlabel=None,
        ylabel=None,
        yticks=np.arange(-30, 40, 10),
        xaxis_date_format="%b '%y",
        xaxis_major_locator=matplotlib.dates.MonthLocator((1, 3, 5, 7, 9, 11)),
        font_size=46,
        linewidth=1.8,
        legend_loc=2,
        palette="colorblind",
    )
    ts_lineplot(
        ts_data=values,
        save_pathname=".nox/temp/ts_lineplot5.svg",
        ts_x=dates,
        ts_names=["A", "B", "C", "D"],
        title="Example Time Series",
        xlabel=None,
        ylabel=None,
        yticks=np.arange(-30, 40, 10),
        xaxis_date_format="%b '%y",
        xaxis_major_locator=matplotlib.dates.MonthLocator((1, 3, 5, 7, 9, 11)),
        linewidth=1.5,
        legend_loc=2,
        palette="colorblind",
        font_size=20,
        height=4.4,
        aspect=1.8,
    )
