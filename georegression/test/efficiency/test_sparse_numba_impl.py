import time

from numba import njit
from scipy.sparse import  csr_array
import numpy as np
from scipy.spatial.distance import pdist, cdist
from functools import reduce

# @njit()
def second_neighbour_matrix(neighbour_csr: csr_array):
    second_neighbour_indices_list = []
    for row_index in range(neighbour_csr.shape[0]):
        column_indices = neighbour_csr.indices[neighbour_csr.indptr[row_index]:neighbour_csr.indptr[row_index + 1]]
        second_neighbour_final = []
        for column_index in column_indices:
            second_neighbour_indices = neighbour_csr.indices[
                neighbour_csr.indptr[column_index] : neighbour_csr.indptr[column_index + 1]
            ]
            # second_neighbour_final = np.union1d(second_neighbour_final, second_neighbour_indices)
            second_neighbour_final.append(second_neighbour_indices)
        if len(second_neighbour_final) == 0:
            r = np.array([])
        else:
            r = reduce(np.union1d, second_neighbour_final)
        second_neighbour_indices_list.append(r)

    return second_neighbour_indices_list

@njit()
def union1d_wrapper(arr1, arr2):
    return np.union1d(arr1, arr2)

@njit()
def second_neighbour_matrix_numba(indptr, indices):
    """
    TODO: More deep understanding of the numba is required.

    Args:
        indptr ():
        indices ():

    Returns:

    """

    N = len(indptr) - 1
    second_neighbour_matrix = np.zeros((N, N))
    for row_index in range(N):
        neighbour_indices = indices[indptr[row_index]:indptr[row_index + 1]]
        second_neighbour_indices_union = np.zeros((N,))
        for neighbour_index in neighbour_indices:
            second_neighbour_indices = indices[
                indptr[neighbour_index] : indptr[neighbour_index + 1]
            ]
            for second_neighbour_index in second_neighbour_indices:
                second_neighbour_indices_union[second_neighbour_index] = True

        second_neighbour_matrix[row_index] = second_neighbour_indices_union

    return second_neighbour_matrix


@njit()
def second_neighbour_matrix_numba_2(indptr, indices):
    """
    Return in sparse format.
    """

    indices_list = []
    N = len(indptr) - 1
    # second_neighbour_matrix = np.zeros((N, N))
    for row_index in range(N):
        neighbour_indices = indices[indptr[row_index]:indptr[row_index + 1]]
        second_neighbour_indices_union = np.zeros((N,))
        for neighbour_index in neighbour_indices:
            second_neighbour_indices = indices[
                indptr[neighbour_index] : indptr[neighbour_index + 1]
            ]
            for second_neighbour_index in second_neighbour_indices:
                second_neighbour_indices_union[second_neighbour_index] = True

        second_neighbour_indices_union = np.nonzero(second_neighbour_indices_union)[0]
        indices_list.append(second_neighbour_indices_union)

        # second_neighbour_matrix[row_index] = second_neighbour_indices_union

    return indices_list

def test_second_neighbour_matrix():
    points = np.random.random((10000, 2))
    distance_matrix = cdist(points, points)
    neighbour_matrix = csr_array(distance_matrix > 0.95)

    t1 = time.time()
    # r = second_neighbour_matrix(neighbour_matrix)
    t2 = time.time()
    print(t2 - t1)

    r = second_neighbour_matrix_numba(neighbour_matrix.indptr, neighbour_matrix.indices)

    t3 = time.time()
    r = second_neighbour_matrix_numba(neighbour_matrix.indptr, neighbour_matrix.indices)
    t4 = time.time()
    print(t4 - t3)

    r = second_neighbour_matrix_numba_2(neighbour_matrix.indptr, neighbour_matrix.indices)

    t5 = time.time()
    r = second_neighbour_matrix_numba_2(neighbour_matrix.indptr, neighbour_matrix.indices)
    t6 = time.time()
    print(t6 - t5)

    print()


def bool_type():
    m = np.random.random((100, 100)) > 0.5
    s = csr_array(m)
    # TODO: bool type is not fully compressed, as there is duplicated True value in the data array.
    print()


if __name__ == '__main__':
    # bool_type()
    test_second_neighbour_matrix()
    pass