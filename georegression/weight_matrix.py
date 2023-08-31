from time import time
from typing import Union

import dask.array as da
import numpy as np
from dask.graph_manipulation import wait_on
from slab_utils.quick_logger import logger

from georegression.distance_utils import distance_matrices
from georegression.kernel import kernel_function, adaptive_kernel


def calculate_compound_weight_matrix(source_coordinate_vector_list: list[np.ndarray], target_coordinate_vector_list: list[np.ndarray], distance_measure: Union[str, list[str]],
                                     kernel_type: Union[str, list[str]], distance_ratio: Union[float, list[float]] = None, bandwidth: Union[float, list[float]] = None,
                                     neighbour_count: Union[float, list[float]] = None, distance_args: Union[dict, list[dict]] = None) -> np.ndarray:
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

    t_start = time()

    compound_weight_matrix = compound_weight(distance_matrices(
        source_coordinate_vector_list,
        target_coordinate_vector_list,
        distance_measure,
        distance_args,
    ), kernel_type, distance_ratio, bandwidth, neighbour_count)

    logger.debug(f"Time taken to calculate compound weight matrix: {time() - t_start}")

    return compound_weight_matrix


def compound_weight(distance_matrices: list[np.ndarray], kernel_type: Union[str, list[str]], distance_ratio: Union[float, list[float], None] = None, bandwidth: Union[float, list[float], None] = None,
                    neighbour_count: Union[float, list[float], None] = None) -> Union[np.ndarray, da.array]:
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

    # Dimension of the vector list. (Len of the vector list)
    dimension = len(distance_matrices)

    source_size = distance_matrices[0].shape[0]
    target_size = distance_matrices[0].shape[1]

    # Support for dask
    weight_matrix = []

    # Check whether the size of distance matrices are the same.
    if len(set([distance_matrix.shape for distance_matrix in distance_matrices])) != 1:
        raise Exception("Size of distance matrices are not the same")

    # Check whether to use fixed kernel or adaptive kernel
    if bandwidth is None and neighbour_count is None:
        raise Exception(
            "At least one of bandwidth or neighbour count should be provided"
        )

    # Integrate distance or weight.
    if distance_ratio is not None:
        # Integrate distance

        if not isinstance(distance_ratio, list) and dimension != 2:
            raise Exception(
                "Distance ratio list must be provided for dimension larger than 2"
            )

        if isinstance(kernel_type, list):
            raise Exception("Kernel type cannot be list while integrating distance")

        if isinstance(bandwidth, list):
            raise Exception("Bandwidth cannot be list while integrating distance")

        if isinstance(neighbour_count, list):
            raise Exception("Neighbour count cannot be list while integrating distance")


        # TODO: Normalization step should be considered.

        # TODO: More operation, not only addition, should be considered.
        #  Like different distance measurements (replace distance_diff in distance_utils.py).
        #  Or some arithmetic operations like multiplication or division?

        if not isinstance(distance_ratio, list):
            distance_ratio = [1, distance_ratio]

        # Iterate source
        for source_index in range(source_size):
            distances = [
                distance_matrices[dim][source_index, :] for dim in range(dimension)
            ]

            # (n, d) * (d, 1) = (n, 1)
            distance = np.matmul(
                np.array(distances).T, np.array(distance_ratio, ndmin=2).T
            )

            weight = weight_by_distance(distance, kernel_type, bandwidth, neighbour_count)

            weight_matrix.append(weight)

    else:
        # Integrate weight

        if not isinstance(kernel_type, list):
            kernel_type = [kernel_type] * dimension

        if not isinstance(bandwidth, list):
            bandwidth = [bandwidth] * dimension

        if not isinstance(neighbour_count, list):
            neighbour_count = [neighbour_count] * dimension


        # TODO: Also should check the dimension of the parameters.

        weights = []
        for dim in range(dimension):
            if isinstance(distance_matrices[0], da.Array):
                weights.append(
                    wait_on(
                        weight_by_distance(distance_matrices[dim], kernel_type[dim], bandwidth[dim], neighbour_count[dim])
                    )
                )
            else:
                weights.append(
                    weight_by_distance(distance_matrices[dim], kernel_type[dim], bandwidth[dim], neighbour_count[dim])
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

    return weight_matrix_norm


def weight_by_distance(distance_vector, kernel_type, bandwidth=None, neighbour_count=None):
    """
    Using fixed kernel(bandwidth provided) or adaptive kernel(neighbour count provided)
    to calculate the weight based on the distance vector.

    Args:
        distance_vector:
        kernel_type:
        bandwidth:
        neighbour_count:

    Returns:

    """

    if bandwidth is not None and neighbour_count is None:
        weight = kernel_function(distance_vector, bandwidth, kernel_type)
    elif bandwidth is None and neighbour_count is not None:
        weight = adaptive_kernel(distance_vector, neighbour_count, kernel_type)
    else:
        raise Exception(
            "Choose bandwidth for fixed kernel or neighbour count for adaptive kernel"
        )
    return weight
