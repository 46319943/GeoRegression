from time import time
from typing import Union

import dask.array as da
import numpy as np
from dask.graph_manipulation import wait_on
from scipy import sparse
from slab_utils.quick_logger import logger

from georegression.distance_utils import _distance_matrices
from georegression.kernel import kernel_function, adaptive_kernel


def weight_matrix_from_points(
        source_coordinate_vector_list: list[np.ndarray],
        target_coordinate_vector_list: list[np.ndarray] = None,
        distance_measure: Union[str, list[str]] = None,
        kernel_type: Union[str, list[str]] = None,
        distance_ratio: Union[float, list[float]] = None,
        bandwidth: Union[float, list[float]] = None,
        neighbour_count: Union[float, list[float]] = None,
        distance_args: Union[dict, list[dict]] = None
) -> np.ndarray:
    """
    Iterate over each source-target pair to get weight matrix.
    Each row represent each source. Each column represent each target.
    The shape of the matrix is (number of source, number of target).

    Args:
        source_coordinate_vector_list:
        target_coordinate_vector_list:
        distance_measure:
        kernel_type:
        distance_ratio:
        bandwidth:
        neighbour_count:
        distance_args:

    Returns:

    """

    log_str = f"\nWeight Matrix from Points Info:\n"
    log_str += f"Coordinate Dimension: {len(source_coordinate_vector_list)}\n"
    for i, coordinate_vector in enumerate(source_coordinate_vector_list):
        log_str += f"coordinate_vector[{i}].shape: {coordinate_vector.shape}\n"
    logger.debug(log_str)

    t_distance_start = time()
    distance_matrices = _distance_matrices(
        source_coordinate_vector_list,
        target_coordinate_vector_list,
        distance_measure,
        distance_args,
    )
    t_distance_end = time()

    t_kernel_start = time()
    compound_weight_matrix = weight_matrix_from_distance(
        distance_matrices, kernel_type, distance_ratio, bandwidth, neighbour_count
    )
    t_kernel_end = time()

    logger.debug(f"Distance Time: {t_distance_end - t_distance_start}. Kernel Time: {t_kernel_end - t_kernel_start}")

    return compound_weight_matrix


def weight_matrix_from_distance(
        distance_matrices,
        kernel_type: Union[str, list[str]],
        distance_ratio: Union[float, list[float], None] = None,
        bandwidth: Union[float, list[float], None] = None,
        neighbour_count: Union[float, list[float], None] = None,
        distance_matrices_sorted: Union[np.ndarray, da.array] = None
) -> Union[np.ndarray, da.array]:
    """
    Calculate weights for each coordinate vector (e.g. location coordinate vector or time coordinate vector)
    and integrate the weights of each coordinate vector to one weight
    using some arithmetic operations (e.g. add or multiply).

    Or in the reversed order, Integrate the distances of each coordinate vector and calculate the weight.
    In this case, `distance_ratio` should be provided.

    All the parameters can provide in list form if weights are integrated instead of distance.
    Length of the lists should match the dimension(or length) of the vector list.

    Args:
        one_coordinate_vector_list:
        many_coordinate_vector_list:
        distance_measure:
        kernel_type:
        distance_ratio:
        bandwidth:
        neighbour_count:
        p:

    Returns:

    """

    log_str = f"\nWeight Matrix from Distance Info:\n"
    log_str += f"Distance Dimension: {len(distance_matrices)}\n"
    for i, distance_matrix in enumerate(distance_matrices):
        log_str += f"distance_matrix[{i}].shape: {distance_matrix.shape}\n"
    log_str += (
        f"kernel_type: {kernel_type}\n"
        f"distance_ratio: {distance_ratio}\n"
        f"bandwidth: {bandwidth}\n"
        f"neighbour_count: {neighbour_count}\n"
    )
    logger.debug(log_str)

    # Dimension of the vector list. (Len of the vector list)
    dimension = len(distance_matrices)

    # Check whether the size of distance matrices are the same.
    if len(set([distance_matrix.shape for distance_matrix in distance_matrices])) != 1:
        raise Exception("Size of distance matrices are not the same")

    # Check whether to use fixed kernel or adaptive kernel
    if bandwidth is None and neighbour_count is None:
        raise Exception(
            "At least one of bandwidth or neighbour count should be provided"
        )

    # Merge distance matrices.
    if distance_ratio is not None:
        if not isinstance(distance_ratio, list) and dimension != 2:
            raise Exception(
                "Distance ratio list must be provided for dimension larger than 2"
            )

        # TODO: Normalization step should be considered.

        # TODO: More operation, not only addition, should be considered.
        #  Like different distance measurements (replace distance_diff in distance_utils.py).
        #  Or some arithmetic operations like multiplication or division?

        if not isinstance(distance_ratio, list) and dimension == 2:
            distance_ratio = [1, distance_ratio]
        else:
            if len(distance_ratio) != dimension:
                raise Exception(
                    "Length of distance ratio list must match the dimension of the vector list"
                )

        distance_matrices_temp = []
        for dim in range(dimension):
            distance_matrices_temp[dim] = distance_matrices[dim] * distance_ratio[dim]

        distance_matrix = np.sum(distance_matrices_temp, axis=0)
        distance_matrices = [distance_matrix]

        dimension = 1

    # Then, calculate weight matrices and merge.

    # Also should check the dimension of the parameters if it is already a list?
    if not isinstance(kernel_type, list):
        kernel_type = [kernel_type] * dimension

    if not isinstance(bandwidth, list):
        bandwidth = [bandwidth] * dimension

    if not isinstance(neighbour_count, list):
        neighbour_count = [neighbour_count] * dimension

    if not isinstance(distance_matrices_sorted, list):
        distance_matrices_sorted = [distance_matrices_sorted] * dimension

    weights = []
    for dim in range(dimension):
        if isinstance(distance_matrices[0], da.Array):
            weights.append(
                # Need to wait on?
                weight_by_distance(
                    distance_matrices[dim], kernel_type[dim], bandwidth[dim], neighbour_count[dim],
                    distance_matrices_sorted[dim]
                )
            )
        else:
            weights.append(
                weight_by_distance(distance_matrices[dim], kernel_type[dim], bandwidth[dim], neighbour_count[dim], None)
            )

    weights = np.stack(weights)

    # TODO: Not only multiplication? e.g. Addition, minimum, maximum, average
    weight_matrix = np.prod(weights, axis=0)

    # Normalization
    # TODO: More normalization option. The key point is the proportion in a row?

    # default use row normalization
    row_sum = np.sum(weight_matrix, axis=1)
    # for some row with all 0 weight.
    row_sum[row_sum == 0] = 1
    # Notice the axis of division
    weight_matrix_norm = weight_matrix / np.expand_dims(row_sum, 1)

    if isinstance(weight_matrix_norm, da.Array):
        weight_matrix_norm = weight_matrix_norm.map_blocks(sparse.coo_matrix).compute()

    # Stat the non-zero weight ratio
    logger.debug(
        f"Non-zero weight ratio: {np.count_nonzero(weight_matrix_norm) / weight_matrix_norm.size}\n"
        f"Average neighbour count: {np.mean(np.count_nonzero(weight_matrix_norm, axis=1))}"
    )


    return weight_matrix_norm


def weight_by_distance(distance, kernel_type, bandwidth, neighbour_count, distance_sorted):
    """
    Using fixed kernel(bandwidth provided) or adaptive kernel(neighbour count provided)
    to calculate the weight based on the distance vector.

    Args:
        distance:
        kernel_type:
        bandwidth:
        neighbour_count:

    Returns:

    """

    if bandwidth is not None and neighbour_count is None:
        weight = kernel_function(distance, bandwidth, kernel_type)
    elif bandwidth is None and neighbour_count is not None:
        weight = adaptive_kernel(distance, neighbour_count, kernel_type, distance_sorted)
    else:
        raise Exception(
            "Choose bandwidth for fixed kernel or neighbour count for adaptive kernel"
        )
    return weight
