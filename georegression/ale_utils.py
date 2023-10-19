"""
References: https://github.com/SeldonIO/alibi/blob/master/alibi/explainers/ale.py
"""

from functools import partial
from typing import Callable, Tuple

import numpy as np


def get_quantiles(values: np.ndarray, num_quantiles: int = 11, interpolation='linear') -> np.ndarray:
    """
    Calculate quantiles of values in an array.

    Parameters
    ----------
    values
        Array of values.
    num_quantiles
        Number of quantiles to calculate.

    Returns
    -------
    Array of quantiles of the input values.

    """
    percentiles = np.linspace(0, 100, num=num_quantiles)
    quantiles = np.percentile(values, percentiles, axis=0, interpolation=interpolation)  # type: ignore[call-overload]
    return quantiles


def bisect_fun(fun: Callable, target: float, lo: int, hi: int) -> int:
    """
    Bisection algorithm for function evaluation with integer support.

    Assumes the function is non-decreasing on the interval `[lo, hi]`.
    Return an integer value v such that for all `x<v, fun(x)<target` and for all `x>=v, fun(x)>=target`.
    This is equivalent to the library function `bisect.bisect_left` but for functions defined on integers.

    Parameters
    ----------
    fun
        A function defined on integers in the range `[lo, hi]` and returning floats.
    target
        Target value to be searched for.
    lo
        Lower bound of the domain.
    hi
        Upper bound of the domain.

    Returns
    -------
    Integer index.

    """
    while lo < hi:
        mid = (lo + hi) // 2
        if fun(mid) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def minimum_satisfied(values: np.ndarray, min_bin_points: int, n: int) -> int:
    """
    Calculates whether the partition into bins induced by `n` quantiles
    has the minimum number of points in each resulting bin.

    Parameters
    ----------
    values
        Array of feature values.
    min_bin_points
        Minimum number of points each discretized interval needs to contain.
    n
        Number of quantiles.

    Returns
    -------
    Integer encoded boolean with 1 - each bin has at least `min_bin_points` and 0 otherwise.

    """
    q = np.unique(get_quantiles(values, num_quantiles=n))
    indices = np.searchsorted(q, values, side='left')
    indices[indices == 0] = 1
    interval_n = np.bincount(indices)
    return int(np.all(interval_n[1:] > min_bin_points))


def adaptive_grid(values: np.ndarray, min_bin_points: int = 1) -> Tuple[np.ndarray, int]:
    """
    Find the optimal number of quantiles for the range of values so that each resulting bin
    contains at least `min_bin_points`. Uses bisection.

    Parameters
    ----------
    values
        Array of feature values.
    min_bin_points
        Minimum number of points each discretized interval should contain to ensure more precise
        ALE estimation.

    Returns
    -------
    q
        Unique quantiles.
    num_quantiles
        Number of non-unique quantiles the feature array was subdivided into.

    Notes
    -----
    This is a heuristic procedure since the bisection algorithm is applied
    to a function which is not monotonic. This will not necessarily find the
    maximum number of bins the interval can be subdivided into to satisfy
    the minimum number of points in each resulting bin.
    """

    # function to bisect
    def minimum_not_satisfied(values: np.ndarray, min_bin_points: int, n: int) -> int:
        """
        Logical not of `minimum_satisfied`, see function for parameter information.
        """
        return 1 - minimum_satisfied(values, min_bin_points, n)

    fun = partial(minimum_not_satisfied, values, min_bin_points)

    # bisect
    num_quantiles = bisect_fun(fun=fun, target=0.5, lo=0, hi=len(values)) - 1
    q = np.unique(get_quantiles(values, num_quantiles=num_quantiles))

    return q, num_quantiles
