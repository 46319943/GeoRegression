import time

import numpy as np
from numba import njit, prange
from scipy.sparse import csr_matrix, csr_array, lil_array
from scipy.spatial.distance import cdist


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