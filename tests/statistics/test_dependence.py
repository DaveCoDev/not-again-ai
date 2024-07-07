from typing import Any

import numpy as np
import numpy.typing as npt

from not_again_ai.statistics.dependence import pearson_correlation, pred_power_score_classification


def _example_1(rs: np.random.RandomState) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Example 1 - x is mostly predictive of y.
    x is categorical (strings), y is binary, both are numpy arrays
    """
    x0 = rs.choice(["a"], 200)
    y0 = rs.choice([0, 1], 200, p=[0.9, 0.1])

    x1 = rs.choice(["b"], 300)
    y1 = rs.choice([0, 1], 300, p=[0.1, 0.9])

    x = np.concatenate([x0, x1])
    y = np.concatenate([y0, y1])
    return (x, y)


def _example_2() -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Example 2 - x completely predicts y"""
    x = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y = np.array(["a", "a", "b", "b", "b", "b", "b", "b", "b", "b"])
    return (x, y)


def _example_3() -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Example 3 - y contains all of the same values"""
    x = np.array([0, 0, 0, 0, 0, 0])
    y = np.array([3, 3, 3, 3, 3, 3])
    return (x, y)


def _example_4(rs: np.random.RandomState) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Example 4 - y is multi-class, x is numeric"""
    x0 = rs.normal(3, 0.1, 100)
    yo = rs.choice([0, 1, 2], 100, p=[0.9, 0.05, 0.05])

    x1 = rs.normal(2, 0.1, 100)
    y1 = rs.choice([0, 1, 2], 100, p=[0.05, 0.9, 0.05])

    x2 = rs.normal(1, 0.1, 100)
    y2 = rs.choice([0, 1, 2], 100, p=[0.05, 0.05, 0.9])

    x = np.concatenate([x0, x1, x2])
    y = np.concatenate([yo, y1, y2])
    return (x, y)


def _example_5(rs: np.random.RandomState) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Example 5 - x is not predictive of y (random noise)"""
    x = rs.choice(["a", "b"], 500)
    y = rs.choice([0, 1], 500)
    return (x, y)


def _example_6(rs: np.random.RandomState) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Example 6 - Both variables fully random: Correlation should be 0"""
    x = rs.randn(500)
    y = rs.randn(500)
    return (x, y)


def _example_7(rs: np.random.RandomState) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Example 7 - y = x^2 + noise"""
    x = (rs.rand(500) * 4) - 2
    y = x**2 + (rs.randn(500) * 0.2)
    return (x, y)


def _example_8(rs: np.random.RandomState) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Example 8 - y = x + noise, correlation should be  high"""
    x = rs.randn(500)
    e = rs.randn(500) * 0.2
    y = x + e
    return (x, y)


def test_pearson_correlation() -> None:
    rs = np.random.RandomState(365)

    x, y = _example_1(rs)
    res = pearson_correlation(x, y, is_x_categorical=True, is_y_categorical=True)
    assert res > 0.333

    x, y = _example_2()
    res = pearson_correlation(x, y, is_x_categorical=True, is_y_categorical=True)
    assert np.isclose(res, 1, atol=1e-6)

    x, y = _example_3()
    res = pearson_correlation(x, y, is_x_categorical=True, is_y_categorical=True, print_diagnostics=True)
    assert res == 1

    x, y = _example_4(rs)
    res = pearson_correlation(x, y, is_x_categorical=False, is_y_categorical=True)
    assert res > 0.333

    x, y = _example_5(rs)
    res = pearson_correlation(x, y, is_x_categorical=True, is_y_categorical=True)
    assert res < 0.333

    x, y = _example_6(rs)
    res = pearson_correlation(x, y)
    assert res < 0.333

    x, y = _example_7(rs)
    res = pearson_correlation(x, y)
    assert res >= 0

    x, y = _example_8(rs)
    res = pearson_correlation(x, y)
    assert res > 0.5


def test_pred_power_score_classification() -> None:
    rs = np.random.RandomState(365)

    x, y = _example_1(rs)
    res = pred_power_score_classification(x, y)
    assert res > 0.333

    x, y = _example_2()
    res = pred_power_score_classification(x, y)
    assert res == 1

    x, y = _example_3()
    res = pred_power_score_classification(x, y, print_diagnostics=True)
    assert res == 1

    x, y = _example_4(rs)
    res = pred_power_score_classification(x, y)
    assert res > 0.333

    x, y = _example_5(rs)
    res = pred_power_score_classification(x, y, cv_splits=10)
    assert res < 0.333
