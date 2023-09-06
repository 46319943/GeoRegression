import time
from functools import reduce

import numpy as np
from numba import njit,prange, boolean
from scipy.sparse import csr_array
from scipy.spatial.distance import cdist


def second_neighbour_matrix(neighbour_csr: csr_array):
    second_neighbour_indices_list = []
    for row_index in range(neighbour_csr.shape[0]):
        column_indices = neighbour_csr.indices[neighbour_csr.indptr[row_index]:neighbour_csr.indptr[row_index + 1]]
        second_neighbour_final = []
        for column_index in column_indices:
            second_neighbour_indices = neighbour_csr.indices[
                                       neighbour_csr.indptr[column_index]: neighbour_csr.indptr[column_index + 1]
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
def second_neighbour_matrix_numba(indptr, indices):
    """
    TODO: More deep understanding of the numba is required.

    TODO: Add reduce parallel
    TODO: Add loop parallel

    Args:
        indptr ():
        indices ():

    Returns:

    """

    N = len(indptr) - 1
    # Numba type instead of numpy type should be provided here.
    second_neighbour_matrix = np.zeros((N, N), dtype=boolean)
    for row_index in range(N):
        neighbour_indices = indices[indptr[row_index]:indptr[row_index + 1]]
        second_neighbour_indices_union = np.zeros((N,))
        for neighbour_index in neighbour_indices:
            second_neighbour_indices = indices[
                                       indptr[neighbour_index]: indptr[neighbour_index + 1]
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
    for row_index in range(N):
        neighbour_indices = indices[indptr[row_index]:indptr[row_index + 1]]
        second_neighbour_indices_union = np.zeros((N,))
        for neighbour_index in neighbour_indices:
            second_neighbour_indices = indices[
                                       indptr[neighbour_index]: indptr[neighbour_index + 1]
                                       ]
            for second_neighbour_index in second_neighbour_indices:
                second_neighbour_indices_union[second_neighbour_index] = True

        second_neighbour_indices_union = np.nonzero(second_neighbour_indices_union)[0]
        indices_list.append(second_neighbour_indices_union)

    return indices_list


@njit(parallel=True)
def second_neighbour_matrix_numba_loop(indptr, indices):
    """
    Return in sparse format.
    """

    N = len(indptr) - 1
    # Manually create the list with specified length to avoid parallel Mutating error.
    indices_list = [np.empty(0, dtype=np.int64)] * N
    for row_index in prange(N):
        neighbour_indices = indices[indptr[row_index]:indptr[row_index + 1]]
        second_neighbour_indices_union = np.zeros((N,))
        for neighbour_index in neighbour_indices:
            second_neighbour_indices = indices[
                                       indptr[neighbour_index]: indptr[neighbour_index + 1]
                                       ]
            for second_neighbour_index in second_neighbour_indices:
                second_neighbour_indices_union[second_neighbour_index] = True

        second_neighbour_indices_union = np.nonzero(second_neighbour_indices_union)[0]
        indices_list[row_index] = second_neighbour_indices_union

    return indices_list


@njit(parallel=True)
def second_neighbour_matrix_numba_loop_2(indptr, indices):
    """
    Return in sparse format.
    """

    N = len(indptr) - 1
    # Manually create the list with specified length to avoid parallel Mutating error.
    indices_list = [np.empty(0, dtype=np.int64)] * N
    for row_index in prange(N):
        neighbour_indices = indices[indptr[row_index]:indptr[row_index + 1]]
        second_neighbour_indices_union = np.zeros((N,))
        for neighbour_index in neighbour_indices:
            second_neighbour_indices = indices[
                                       indptr[neighbour_index]: indptr[neighbour_index + 1]
                                       ]

            # TODO: Consider using the np.union1d here? Unknown and variate length may cause performance issue.
            for second_neighbour_index in second_neighbour_indices:
                second_neighbour_indices_union[second_neighbour_index] = True

        second_neighbour_indices_union = np.nonzero(second_neighbour_indices_union)[0]
        indices_list[row_index] = second_neighbour_indices_union

    return indices_list

def test_second_neighbour_matrix():
    points = np.random.random((10000, 2))
    distance_matrix = cdist(points, points)
    neighbour_matrix = csr_array(distance_matrix > 0.8)

    t1 = time.time()
    # r = second_neighbour_matrix(neighbour_matrix)
    t2 = time.time()
    print(t2 - t1)

    # second_neighbour_matrix_numba(neighbour_matrix.indptr, neighbour_matrix.indices)
    t3 = time.time()
    # r = second_neighbour_matrix_numba(neighbour_matrix.indptr, neighbour_matrix.indices)
    t4 = time.time()
    print(t4 - t3)

    # second_neighbour_matrix_numba_2(neighbour_matrix.indptr, neighbour_matrix.indices)
    t5 = time.time()
    # r = second_neighbour_matrix_numba_2(neighbour_matrix.indptr, neighbour_matrix.indices)
    t6 = time.time()
    print(t6 - t5)

    second_neighbour_matrix_numba_loop(neighbour_matrix.indptr, neighbour_matrix.indices)
    t7 = time.time()
    r = second_neighbour_matrix_numba_loop(neighbour_matrix.indptr, neighbour_matrix.indices)
    t8 = time.time()
    print(t8 - t7)

    # second_neighbour_matrix_numba_loop_2(neighbour_matrix.indptr, neighbour_matrix.indices)
    t9 = time.time()
    # r = second_neighbour_matrix_numba_loop_2(neighbour_matrix.indptr, neighbour_matrix.indices)
    t10 = time.time()
    print(t10 - t9)

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
