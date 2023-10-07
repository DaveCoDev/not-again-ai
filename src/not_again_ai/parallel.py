from collections.abc import Callable
from multiprocessing.pool import ThreadPool
from typing import Any


def embarrassingly_parallel(
    func: Callable[..., Any],
    args_list: tuple[tuple[Any, ...], ...] | None,
    kwargs_list: list[dict[str, Any]] | None = None,
    num_processes: int = 1,
) -> list[Any]:
    """Call multiple functions in parallel providing either positional arguments, keyword arguments,
        or both. Return the function returns in a list ordered by order of the input arguments.

    If both are provided, positional and keyword arguments must be aligned in the same order
    and each list must be the same length.

    Args:
        func (Callable[..., Any]): Any function
        args_list (Optional[tuple[tuple[Any, ...], ...]]): A tuple of tuples each of positional arguments.
        kwargs_list (Optional[list[dict[str, Any]]], optional): A list of dictionaries containing keyword arguments. Defaults to None.
        num_processes (int, optional): Number of parallel processors to use. Defaults to 1.

    Raises:
        ValueError: If positional and keyword arguments are not aligned in the
            same order or if the lists are not the same length.
        ValueError: If neither positional nor keyword arguments are provided.

    Returns:
        list[Any]: list of the returns of each function call in order of the args_list or kwargs_list.
    """

    pool = ThreadPool(processes=num_processes)
    results = {}
    if (args_list is not None) and (kwargs_list is None):
        for idx, args in enumerate(args_list):
            results[idx] = pool.apply_async(func, args)
    elif (args_list is None) and (kwargs_list is not None):
        for idx, kwargs in enumerate(kwargs_list):
            results[idx] = pool.apply_async(func, kwds=kwargs)
    elif (args_list is not None) and (kwargs_list is not None):
        # in this case args_list and kwargs_list must be of the same length
        if len(args_list) == len(kwargs_list):
            for idx, (args, kwargs) in enumerate(zip(args_list, kwargs_list, strict=True)):
                results[idx] = pool.apply_async(func, args, kwargs)
        else:
            pool.close()
            pool.terminate()
            raise ValueError("args_list and kwargs_list must be of the same length")
    else:
        pool.close()
        pool.terminate()
        raise ValueError("either args_list or kwargs_list must be provided")

    return_results = []
    for _, res in results.items():
        return_results.append(res.get())

    pool.close()
    pool.terminate()
    return return_results


def embarrassingly_parallel_simple(funcs: list[Callable[..., Any]], num_processes: int = 1) -> list[Any]:
    """Executes the given functions in parallel and returns the results in the same order as the funcs were provided.

    Args:
        funcs (list[Callable[..., Any]]): A list of any functions that take no arguments.
        num_processes (int, optional): Number of parallel processors to use. Defaults to 1.

    Returns:
        list[Any]: list of the returns of each function call in order of the provided funcs.
    """

    pool = ThreadPool(processes=num_processes)
    results = {}
    for idx, func in enumerate(funcs):
        results[idx] = pool.apply_async(func)

    return_results = []
    for _, res in results.items():
        return_results.append(res.get())

    pool.close()
    pool.terminate()
    return return_results
