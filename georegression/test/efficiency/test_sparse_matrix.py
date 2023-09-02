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


def test_sparse_operation():
    arr = (np.random.random((10, 10)) - 0.8) > 0
    # TODO: Consider using lil_array for modifying the array.
    arr = csr_matrix(arr, dtype=bool)

    # Analogy of nonzere. Ref:https://numpy.org/devdocs/user/basics.indexing.html#boolean-array-indexing
    second_nei = np.sum(arr[arr[0].nonzero()[1]], axis=0) > 0
    print(second_nei)

    # Set value
    second_neighbour_matrix = csr_matrix((10, 10), dtype=bool)
    second_neighbour_matrix[0, second_nei.nonzero()[1]] = True
    print(second_neighbour_matrix)

    # Index array by sparse matrix
    np.random.random((10, 10))[second_neighbour_matrix[0].nonzero()]


@njit()
def second_order_neighbour(neighbour_matrix: csr_matrix):
    second_order_matrix = np.empty_like(neighbour_matrix)
    for i in prange(neighbour_matrix.shape[0]):
        second_order_matrix[i] = np.sum(neighbour_matrix[neighbour_matrix[i]], axis=0)
    return second_order_matrix

def second_order_neighbour_sparse(neighbour_matrix: csr_matrix):
    second_order_matrix = lil_array((neighbour_matrix.shape[0], neighbour_matrix.shape[1]), dtype=bool)
    for i in prange(neighbour_matrix.shape[0]):
        second_order_matrix[i] = np.sum(neighbour_matrix[neighbour_matrix[[i], :].nonzero()[1], :], axis=0) > 0

    return second_order_matrix

def test_njit():
    pass

def test_second_order_neighbour():
    points = np.random.random((10000, 2))
    distance_matrix = cdist(points, points)
    neighbour_matrix = distance_matrix > 0.95

    m = neighbour_matrix
    s = csr_array(m)
    t1 = time.time()
    # r = second_order_neighbour_sparse(s)
    t2 = time.time()
    print(t2 - t1)

    second_order_neighbour(m)

    t3 = time.time()
    second_order_neighbour(m)
    t4 = time.time()
    print(t4 - t3)

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
    # test_sparse_operation()
    # test_compatibility()
    # test_stacking_compatibility()
    test_second_order_neighbour()
