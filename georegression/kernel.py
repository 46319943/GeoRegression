import math
from typing import Union

import numpy as np

KERNEL_TYPE_ENUM = ['linear', 'uniform', 'gaussian', 'exponential', 'boxcar', 'bisquare', 'tricube']


def kernel_function(distance_vector: np.ndarray, bandwidth: float, kernel_type: str) -> np.ndarray:
    """
    Using kernel function to calculate the weight

    Args:
        distance_vector: The distances to be weighted by kernel function.

        bandwidth: parameter of the kernel function for specifying the decreasing level.
         For compact supported kernel, weight will be 0 if distance is larger than the bandwidth.

        kernel_type:

    Returns:
        np.ndarray: weight without normalization

    """

    # Only Box-Car kernel would be applied to zero bandwidth
    if bandwidth == 0:
        kernel_type = 'boxcar'

    # Distance divided by bandwidth. Also consider the zero bandwidth
    normalize_distance = distance_vector / bandwidth if bandwidth else np.zeros_like(distance_vector)

    # Continuous kernel
    if kernel_type == 'uniform':
        weight = np.ones_like(normalize_distance)
    elif kernel_type == 'gaussian':
        weight = np.exp(-0.5 * normalize_distance ** 2)
    elif kernel_type == 'exponential':
        weight = np.exp(-0.5 * np.abs(normalize_distance))
    # Compact supported kernel
    elif kernel_type == 'linear':
        weight = 1 - normalize_distance
    elif kernel_type == 'boxcar':
        weight = np.ones_like(normalize_distance)
    elif kernel_type == 'bisquare':
        weight = (1 - normalize_distance ** 2) ** 2
    elif kernel_type == 'tricube':
        weight = (1 - np.abs(normalize_distance) ** 3) ** 3
    else:
        raise Exception('Unsupported kernel')

    # compact support
    if kernel_type in ['linear', 'boxcar', 'bisquare', 'tricube']:
        weight[distance_vector > bandwidth] = 0

    return weight


def adaptive_bandwidth(distance_vector: np.ndarray, neighbour_count: Union[int, float],
                       midpoint: bool = False) -> float:
    """
    Find the bandwidth to include the specified number of neighbour.

    Args:
        distance_vector: The distances to calculate the adaptive bandwidth.
        neighbour_count: Number of the neighbour to include by the bandwidth.
         Use float to specify the percentage of the neighbour to include.
        midpoint: Whether extend the bandwidth to the midpoint of the neighbours.

    Returns:
        float: return the distance to the K nearest neighbour
    """

    # Duplicated coordinate considered as a single neighbour
    distance_unique = np.unique(distance_vector)

    # Convert percentage to absolute neighbour count
    if isinstance(neighbour_count, float):
        neighbour_count = math.ceil(distance_unique.shape[0] * neighbour_count)

    if neighbour_count <= 0:
        raise Exception('Invalid neighbour count')

    # Limit neighbour to valid count
    if neighbour_count > distance_unique.shape[0]:
        neighbour_count = distance_unique.shape[0]

    bandwidth = np.partition(distance_unique, neighbour_count - 1)[neighbour_count - 1]

    # Extend the bandwidth to the midpoint of the current and next neighbout.
    if midpoint:
        bandwidth_plus = np.partition(distance_unique, neighbour_count)[neighbour_count]
        bandwidth = (bandwidth + bandwidth_plus) / 2

    return bandwidth


def adaptive_kernel(distance_vector: np.ndarray, neighbour_count: Union[int, float], kernel_type: str,
                    midpoint: bool = False) -> np.ndarray:
    """
    Deduce the bandwidth from the neighbour count and calculate weight using kernel function.

    Args:
        distance_vector:
        neighbour_count:
        kernel_type:
        midpoint:

    Returns:

    """

    bandwidth = adaptive_bandwidth(distance_vector, neighbour_count, midpoint)
    return kernel_function(distance_vector, bandwidth, kernel_type)
