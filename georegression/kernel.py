import math
from typing import Union

import dask.array as da
import numpy as np

from slab_utils.quick_logger import logger

KERNEL_TYPE_ENUM = [
    "linear",
    "uniform",
    "gaussian",
    "exponential",
    "boxcar",
    "bisquare",
    "tricube",
]


def kernel_function(
    distance: np.ndarray,
    bandwidth: Union[float, list[float], np.ndarray],
    kernel_type: str,
) -> Union[np.ndarray, da.Array]:
    """
    Using kernel function to calculate the weight

    Args:
        distance: The distances to be weighted by kernel function. In vector or matrix form.

        bandwidth: parameter of the kernel function for specifying the decreasing level.
         For compact supported kernel, weight will be 0 if distance is larger than the bandwidth.

        kernel_type:

    Returns:
        np.ndarray: weight without normalization

    """

    # Reshape bandwidth to have proper broadcasting in division. It has been done in previous step.
    normalize_distance = distance / bandwidth
    if isinstance(normalize_distance, da.Array):
        # TODO: Why dask array do not support keyword argument for this method? It can be easily implemented with map_blocks
        # da.map_blocks(np.nan_to_num, normalize_distance, copy=False, nan=0.0, posinf=np.inf)
        normalize_distance = da.nan_to_num(normalize_distance, False, 0.0, np.inf)
    else:
        np.nan_to_num(normalize_distance, copy=False, nan=0.0, posinf=np.inf)

    # Continuous kernel
    if kernel_type == "uniform":
        weight = np.ones_like(normalize_distance)
    elif kernel_type == "gaussian":
        weight = np.exp(-0.5 * normalize_distance**2)
    elif kernel_type == "exponential":
        weight = np.exp(-0.5 * np.abs(normalize_distance))

    # Compact supported kernel
    elif kernel_type == "linear":
        weight = 1 - normalize_distance
    elif kernel_type == "boxcar":
        weight = np.ones_like(normalize_distance)
    elif kernel_type == "bisquare":
        # Optimize for dask array
        if isinstance(normalize_distance, da.Array):
            normalize_distance[normalize_distance > 1] = 1
            weight = (1 - normalize_distance**2) ** 2

            # It will never be negative!!!
            # weight[weight < 0] = 0

            return weight

        weight = (1 - normalize_distance**2) ** 2
    elif kernel_type == "tricube":
        weight = (1 - np.abs(normalize_distance) ** 3) ** 3
    else:
        raise Exception("Unsupported kernel")

    # compact support
    if kernel_type in ["linear", "boxcar", "bisquare", "tricube"]:
        weight[distance > bandwidth] = 0

    return weight


def adaptive_bandwidth(distance: np.ndarray, neighbour_count: Union[int, float], distance_sorted: da.array) -> float:
    """
    Find the bandwidth to include the specified number of neighbour.

    Args:
        distance: The distances to calculate the adaptive bandwidth. In vector or matrix form.
        neighbour_count: Number of the neighbour to include by the bandwidth.
         Use float to specify the percentage of the neighbour to include.

    Returns:
        float: return the distance to the K nearest neighbour
    """

    if isinstance(distance, da.Array):
        # Support for dask array
        # Duplicated coordinate is not supported
        N = distance.shape[0]

        if distance_sorted is not None:
            # Must be the sorted distance matrix
            if isinstance(neighbour_count, float):
                neighbour_count = math.ceil(N * neighbour_count)

            if neighbour_count > N:
                raise Exception("Invalid neighbour count")

            bandwidth = distance_sorted[:, [neighbour_count - 1]]
            return bandwidth
        else:
            logger.warning('Sorted distance matrix is not provided, requiring longer time to perform quantile function')

            bandwidth = distance.map_blocks(
                np.quantile,
                neighbour_count,
                axis=1,
                keepdims=True,
                drop_axis=1,
                new_axis=1,
            )
            return bandwidth

    if neighbour_count <= 0:
        raise Exception("Invalid neighbour count")

    if isinstance(neighbour_count, float):
        # percentile call the partition function internally,
        bandwidth = np.quantile(distance, neighbour_count, axis=1, keepdims=True, method='median_unbiased')
    elif isinstance(neighbour_count, int):
        bandwidth = np.partition(distance, neighbour_count - 1)[:, [neighbour_count - 1]]
    else:
        raise Exception("Invalid neighbour count")

    return bandwidth


def adaptive_kernel(
        distance: np.ndarray,
        neighbour_count: Union[int, float],
        kernel_type: str,
        distance_sorted: da.array=None
) -> np.ndarray:
    """
    Deduce the bandwidth from the neighbour count and calculate weight using kernel function.

    Args:
        distance:
        neighbour_count:
        kernel_type:

    Returns:

    """

    bandwidth = adaptive_bandwidth(distance, neighbour_count, distance_sorted)
    return kernel_function(distance, bandwidth, kernel_type)
