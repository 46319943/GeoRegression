from time import time
from typing import Union

import numpy as np
from joblib import delayed, Parallel

from slab_utils.quick_logger import logger
from georegression.distance_utils import calculate_distance_one_to_many
from georegression.kernel import kernel_function, adaptive_bandwidth, adaptive_kernel

# minimum count of entities to make computation paralleled.
PARALLEL_MINIMUM_COUNT = 5000


def calculate_compound_weight_matrix(source_coordinate_vector_list: list[np.ndarray],
                                     target_coordinate_vector_list: list[np.ndarray],
                                     distance_measure: Union[str, list[str]],
                                     kernel_type: Union[str, list[str]],
                                     distance_ratio: Union[float, list[float]] = None,
                                     bandwidth: Union[float, list[float]] = None,
                                     neighbour_count: Union[float, list[float]] = None,
                                     midpoint: Union[bool, list[bool]] = None,
                                     p: Union[float, list[float]] = None
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
        midpoint:
        p:

    Returns:

    """

    t_start = time()
    logger.debug(
        f'Weight matrix calculate begin {t_start}. '
        f'{len(source_coordinate_vector_list)} Vectors, '
        f'{len(source_coordinate_vector_list[0])} Sources, '
        f'{len(target_coordinate_vector_list[0])} Targets'
    )

    weight_list = []

    # Parallel the computation for large data
    if len(source_coordinate_vector_list[0]) >= PARALLEL_MINIMUM_COUNT:
        parallel = True
    else:
        parallel = False

    if parallel:
        parallel_task = (
            delayed(calculate_compound_weight)(one_coordinate_vector_list, target_coordinate_vector_list,
                                               distance_measure, kernel_type, distance_ratio,
                                               bandwidth, neighbour_count,
                                               midpoint, p)
            for one_coordinate_vector_list in zip(*source_coordinate_vector_list)
        )
        weight_list = Parallel(n_jobs=-1)(parallel_task)
    else:
        for one_coordinate_vector_list in zip(*source_coordinate_vector_list):
            weight = calculate_compound_weight(one_coordinate_vector_list, target_coordinate_vector_list,
                                               distance_measure, kernel_type, distance_ratio,
                                               bandwidth, neighbour_count,
                                               midpoint, p)
            weight_list.append(weight)
    weight_matrix = np.vstack(weight_list)

    # Normalization

    # TODO: More normalization option. The key point is the proportion in a row?

    # default use row normalization
    row_sum = np.sum(weight_matrix, axis=1)
    # for some row with all 0 weight.
    row_sum[row_sum == 0] = 1
    # Notice the axis of division
    weight_matrix_norm = weight_matrix / np.expand_dims(row_sum, 1)

    t_end = time()
    logger.debug(f'Weight matrix calculate end {t_end}. Matrix shape {weight_matrix_norm.shape}')

    return weight_matrix_norm


def calculate_compound_weight(one_coordinate_vector_list: list[np.ndarray],
                              many_coordinate_vector_list: list[np.ndarray],
                              distance_measure: Union[str, list[str]],
                              kernel_type: Union[str, list[str]],
                              distance_ratio: Union[float, list[float]] = None,
                              bandwidth: Union[float, list[float]] = None,
                              neighbour_count: Union[float, list[float]] = None,
                              midpoint: Union[bool, list[bool]] = None,
                              p: Union[float, list[float]] = None
                              ) -> np.ndarray:
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
        midpoint:
        p:

    Returns:

    """

    # Dimension of the vector list. (Len of the vector list)
    dimension = len(one_coordinate_vector_list)
    if len(many_coordinate_vector_list) != dimension:
        raise Exception('Unmatched vector list dimension')

    # Check whether to use fixed kernel or adaptive kernel
    if bandwidth is None and neighbour_count is None:
        raise Exception('At least one of bandwidth or neighbour count should be provided')

    # For distance measurement, expand the single value to list with corresponding dimension.
    if not isinstance(distance_measure, list):
        distance_measure = [distance_measure] * dimension
    if not isinstance(p, list):
        p = [p] * dimension

    # Calculate the distance vector for each coordinate vector
    distance_vector_list = []
    for dimension_index in range(dimension):
        distance_vector = calculate_distance_one_to_many(
            one_coordinate_vector_list[dimension_index], many_coordinate_vector_list[dimension_index],
            distance_measure[dimension_index], p[dimension_index]
        )
        distance_vector_list.append(distance_vector)

    # Integrate distance or weight.
    if distance_ratio is not None:
        # Integrate distance

        if not isinstance(distance_ratio, list) and dimension != 2:
            raise Exception('Distance ratio list must be provided for dimension larger than 2')

        if isinstance(kernel_type, list):
            raise Exception('Kernel type cannot be list while integrating distance')

        if isinstance(bandwidth, list):
            raise Exception('Bandwidth cannot be list while integrating distance')

        if isinstance(neighbour_count, list):
            raise Exception('Neighbour count cannot be list while integrating distance')

        if isinstance(midpoint, list):
            raise Exception('Midpoint cannot be list while integrating distance')

        # TODO: Normalization step should be considered.

        # TODO: More operation, not only addition, should be considered. 
        #  Like different distance measurements (replace distance_diff in distance_utils.py). 
        #  Or some arithmetic operations like multiplication or division?

        if not isinstance(distance_ratio, list):
            distance_ratio = [1, distance_ratio]

        # (n, 2) * (2, 1) = (n, 1)
        distance_vector = np.matmul(
            np.array(distance_vector_list).T,
            np.array(distance_ratio, ndmin=2).T
        )

        weight = calculate_weight(distance_vector, kernel_type, bandwidth, neighbour_count, midpoint)
        return weight

    else:
        # Integrate weight

        if not isinstance(kernel_type, list):
            kernel_type = [kernel_type] * dimension

        if not isinstance(bandwidth, list):
            bandwidth = [bandwidth] * dimension

        if not isinstance(neighbour_count, list):
            neighbour_count = [neighbour_count] * dimension

        if not isinstance(midpoint, list):
            midpoint = [midpoint] * dimension

        # TODO: Also should check the dimension of the parameters.

        # Calculate the weights of each coordinate vector
        weight_list = []
        for dimension_index in range(dimension):
            weight = calculate_weight(distance_vector_list[dimension_index],
                                      kernel_type[dimension_index],
                                      bandwidth[dimension_index],
                                      neighbour_count[dimension_index],
                                      midpoint[dimension_index])
            weight_list.append(weight)

        # TODO: Not only multiplication? e.g. Addition, minimum, maximum, average
        weight = np.prod(weight_list, axis=0)

        return weight


def calculate_weight(distance_vector, kernel_type,
                     bandwidth=None, neighbour_count=None, midpoint=False):
    """
    Using fixed kernel(bandwidth provided) or adaptive kernel(neighbour count provided)
    to calculate the weight based on the distance vector.

    Args:
        distance_vector:
        kernel_type:
        bandwidth:
        neighbour_count:
        midpoint: Whether extend the bandwidth while using adaptive kernel

    Returns:

    """

    if bandwidth is not None and neighbour_count is None:
        weight = kernel_function(distance_vector, bandwidth, kernel_type)
    elif bandwidth is None and neighbour_count is not None:
        weight = adaptive_kernel(distance_vector, neighbour_count, kernel_type, midpoint)
    else:
        raise Exception('Choose bandwidth for fixed kernel or neighbour count for adaptive kernel')
    return weight
