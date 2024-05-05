import numpy as np

from not_again_ai.viz.scatterplot import scatterplot_basic


def test_scatterplot_basic() -> None:
    rs = np.random.RandomState(365)
    x = rs.randn(100)
    e = rs.randn(100) * 0.2
    y = x + e
    scatterplot_basic(x, y, save_pathname=".nox/temp/scatterplot_basic1.png", title="Correlation Chart")
    scatterplot_basic(
        x,
        y,
        save_pathname=".nox/temp/scatterplot_basic2.png",
        title=None,
        xlim=(-10, 10),
        ylim=(-5, 5),
        font_size=36,
        height=15,
        aspect=2.2,
    )
