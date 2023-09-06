import time

import numpy as np
from numba import njit, prange


@njit()
def input_indicator_1(neighbour_matrix):
    indicator_matrix = np.empty_like(neighbour_matrix)
    for i in prange(neighbour_matrix.shape[0]):
        indicator_matrix[i] = np.sum(neighbour_matrix[neighbour_matrix[i]], axis=0)
    return indicator_matrix


@njit()
def input_indicator_2(neighbour_matrix):
    indicator_matrix = []
    for i in prange(neighbour_matrix.shape[0]):
        indicator_matrix.append(np.sum(neighbour_matrix[neighbour_matrix[i]], axis=0))
    return indicator_matrix


@njit()
def input_indicator_3(neighbour_matrix):
    indicator_matrix = []
    for row_neighbour in neighbour_matrix:
        indicator_matrix.append(np.sum(neighbour_matrix[row_neighbour], axis=0))
    return indicator_matrix


def input_indicator_4(neighbour_matrix):
    indicator_matrix = []
    for i in prange(neighbour_matrix.shape[0]):
        indicator_matrix.append(np.sum(neighbour_matrix[neighbour_matrix[i]], axis=0))
    return indicator_matrix


def input_indicator_5(neighbour_matrix):
    indicator_matrix = []
    for row_neighbour in neighbour_matrix:
        indicator_matrix.append(
            neighbour_matrix[row_neighbour].sum(axis=0)
        )
    indicator_matrix = np.array(indicator_matrix)
    return indicator_matrix


def test_weight_indicator():
    estimator_count = 1000

    weight_matrix = np.random.random((estimator_count, estimator_count)) - 0.5
    neighbour_matrix = weight_matrix > 0
    pre_data = (np.random.random((10, 10)) - 0.5) > 0

    # Warm-up for pre-compile
    t0 = time.time()
    input_indicator_1(pre_data)
    input_indicator_2(pre_data)
    input_indicator_3(pre_data)

    t1 = time.time()
    input_indicator_1(neighbour_matrix)
    t2 = time.time()
    input_indicator_2(neighbour_matrix)
    t3 = time.time()
    input_indicator_3(neighbour_matrix)
    t4 = time.time()
    input_indicator_4(neighbour_matrix)
    t5 = time.time()
    input_indicator_5(neighbour_matrix)
    t6 = time.time()

    print()
    print(t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5)
    # 1.3265900611877441
    # 0.0700376033782959 0.07053327560424805 0.07101583480834961 0.4606480598449707 0.4606757164001465
