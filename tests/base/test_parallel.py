import multiprocessing
import random
import time

import pytest

from not_again_ai.base.parallel import embarrassingly_parallel, embarrassingly_parallel_simple


def multby2(x: float, y: float, double: bool = False) -> float:
    time.sleep(random.uniform(0, 1))
    if double:
        return x * y * 2
    else:
        return x * y


def do_something() -> int:
    return 8


def do_something2() -> int:
    return 2


def echo(x: int) -> int:
    time.sleep(random.uniform(0, 1) / 10)
    return x


def test_embarrassingly_parallel() -> None:
    args = ((2, 2), (3, 3), (4, 4))

    result = embarrassingly_parallel(multby2, args, num_processes=multiprocessing.cpu_count())

    total = 0
    for x in result:
        total += x
    assert total == 4 + 9 + 16


def test_embarrassingly_parallel_both() -> None:
    args = ((2, 2), (3, 3), (4, 4))
    kwargs = [{"double": True}, {"double": False}, {"double": True}]

    result = embarrassingly_parallel(multby2, args, kwargs, num_processes=multiprocessing.cpu_count())

    total = 0
    for x in result:
        total += x
    assert total == 8 + 9 + 32


def test_embarrassingly_parallel_kwargs() -> None:
    kwargs = [{"x": 2, "y": 2, "double": True}, {"x": 3, "y": 3, "double": False}, {"x": 4, "y": 4, "double": True}]

    result = embarrassingly_parallel(multby2, None, kwargs, num_processes=multiprocessing.cpu_count())

    total = 0
    for x in result:
        total += x
    assert total == 8 + 9 + 32


def test_embarrassingly_parallel_exceptions() -> None:
    with pytest.raises(ValueError, match="either args_list or kwargs_list must be provided"):
        embarrassingly_parallel(multby2, None, None, num_processes=multiprocessing.cpu_count())

    args = ((2, 2), (3, 3), (4, 4))
    kwargs = [{"double": True}, {"double": False}]
    with pytest.raises(ValueError, match="args_list and kwargs_list must be of the same length"):
        embarrassingly_parallel(multby2, args, kwargs, num_processes=multiprocessing.cpu_count())


def test_embarrassingly_parallel_ordering() -> None:
    args = ((1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,))
    result = embarrassingly_parallel(echo, args, num_processes=3)
    assert result == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_embarrassingly_parallel_simple() -> None:
    result = embarrassingly_parallel_simple([do_something, do_something2], num_processes=2)
    assert result == [8, 2]
