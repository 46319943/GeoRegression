"""
References: https://github.com/SeldonIO/alibi/blob/master/alibi/explainers/ale.py
"""

import numpy as np
import pandas as pd

from georegression.ale_utils import adaptive_grid


def weighted_ale(X, feature, predictor, weights=None, normalize=False, min_bin_points=5):
    fvals, _ = adaptive_grid(X[:, feature], min_bin_points)

    # find which interval each observation falls into
    indices = np.searchsorted(fvals, X[:, feature], side="left")
    indices[indices == 0] = 1  # put the smallest data point in the first interval
    interval_n = np.bincount(indices)  # number of points in each interval

    # predictions for the upper and lower ranges of intervals
    z_low = X.copy()
    z_high = X.copy()
    z_low[:, feature] = fvals[indices - 1]
    z_high[:, feature] = fvals[indices]
    p_low = predictor(z_low)
    p_high = predictor(z_high)

    # finite differences
    p_deltas = p_high - p_low

    # base value, which is the average prediction for the lowest interval
    base_value = np.average(p_low[indices == 1], weights=weights[indices == 1])

    # make a dataframe for averaging over intervals
    concat = np.column_stack((p_deltas, indices, weights))
    df = pd.DataFrame(concat)

    # weighted average for each interval
    avg_p_deltas = df.groupby(1).apply(lambda x: np.average(x[0], weights=x[2])).values

    # accumulate over intervals
    accum_p_deltas = np.cumsum(avg_p_deltas, axis=0)

    # pre-pend 0 for the left-most point
    zeros = np.zeros((1, 1))
    accum_p_deltas = np.insert(accum_p_deltas, 0, zeros, axis=0)

    # center
    if normalize:
        # mean effect, R's `ALEPlot` and `iml` version (approximation per interval)
        # Eq.16 from original paper "Visualizing the effects of predictor variables in black box supervised learning models"
        ale0 = (
                0.5 * (accum_p_deltas[:-1] + accum_p_deltas[1:]) * interval_n[1:]
        ).sum(axis=0)
        ale0 = ale0 / interval_n.sum()

        ale = accum_p_deltas - ale0
    else:
        ale = accum_p_deltas + base_value

    return fvals, ale
