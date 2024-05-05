import numpy as np

from not_again_ai.viz.distributions import univariate_distplot


def test_univariate_distplot() -> None:
    univariate_distplot(data=np.random.normal(size=1000), save_pathname=".nox/temp/distributions_test1.png")
    univariate_distplot(
        data=np.random.normal(size=1000),
        save_pathname=".nox/temp/distributions_test2.png",
        print_summary=False,
        title="Test title",
        xlabel="Test xlabel",
        ylabel="Test ylabel",
        xlim=(-5, 5),
        ylim=(0, 40),
        xticks=None,
        yticks=None,
        bins=200,
        font_size=54,
        height=14,
        aspect=2.2,
    )
    univariate_distplot(
        data=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
        save_pathname=".nox/temp/distributions_test3.png",
        print_summary=False,
        xlim=None,
        ylim=None,
        xticks=np.arange(0, 11, 1),
        yticks=np.arange(0, 5, 1),
        bins=100,
    )
    univariate_distplot(
        data=np.random.beta(a=0.5, b=0.5, size=10000),
        save_pathname=".nox/temp/distributions_test4.svg",
        print_summary=False,
        bins=100,
        title=r"Beta Distribution $\alpha=0.5, \beta=0.5$",
        font_size=18,
        height=3.91,
        aspect=1.8,
    )
