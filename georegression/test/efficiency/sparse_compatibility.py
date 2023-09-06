import time

import numpy as np
from numba import njit, prange
from scipy.sparse import csr_matrix, csr_array, lil_array
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression

from georegression.stacking_model import StackingWeightModel
from georegression.test.data import load_HP
from georegression.weight_matrix import calculate_compound_weight_matrix
from georegression.weight_model import WeightModel


def test_compatibility():
    X, y_true, xy_vector, time = load_HP()

    estimator = WeightModel(
        LinearRegression(),
        "euclidean",
        "bisquare",
        neighbour_count=0.1
    )

    weight_matrix = calculate_compound_weight_matrix(
        [xy_vector, time], [xy_vector, time], "euclidean", "bisquare", None, None, 0.1, None
    )
    sparse_matrix = csr_array(weight_matrix)
    estimator.fit(X, y_true, [xy_vector, time], weight_matrix=sparse_matrix)

    print(estimator.llocv_score_)

    pass

def test_stacking_compatibility():
    X, y_true, xy_vector, time = load_HP()

    estimator = StackingWeightModel(
        LinearRegression(),
        "euclidean",
        "bisquare",
        neighbour_count=0.1,
        neighbour_leave_out_rate=0.3,
    )

    weight_matrix = calculate_compound_weight_matrix(
        [xy_vector, time], [xy_vector, time], "euclidean", "bisquare", None, None, 0.1, None
    )
    sparse_matrix = csr_array(weight_matrix)
    estimator.fit(X, y_true, [xy_vector, time], weight_matrix=sparse_matrix)

    print(estimator.llocv_score_)

    pass

if __name__ == '__main__':
    # test_compatibility()
    test_stacking_compatibility()
