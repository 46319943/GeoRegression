# TODO: https://jaykmody.com/blog/distance-matrices-with-numpy/
# TODO: https://stackoverflow.com/questions/22720864/efficiently-calculating-a-euclidean-distance-matrix-using-numpy

import numpy as np
from numba import njit
from scipy.spatial.distance import pdist, cdist


def distance_matrices(
    source_coords: list[np.ndarray], target_coords=None, metrics=None, args=None
):
    # Check equal length of source and target coordinates
    if target_coords is not None:
        if len(source_coords) != len(target_coords):
            raise Exception("Source and target coordinate length not match")

    dimension = len(source_coords)

    # Check whether the input parameters are lists, and if not, convert them to lists with length equal to the dimension of the vector list.
    if not isinstance(target_coords, list):
        target_coords = [None] * dimension

    if not isinstance(metrics, list):
        metrics = [metrics] * dimension

    if not isinstance(args, list):
        args = [{}] * dimension

    return [
        distance_matrix(
            source_coords[dim],
            target_coords[dim],
            metrics[dim],
            **args[dim],
        )
        for dim in range(dimension)
    ]


def distance_matrix(source_coord, target_coord=None, metric=None, **kwargs):
    # Check equal dimension of source and target coordinates
    if target_coord is not None:
        if source_coord.shape[1] != target_coord.shape[1]:
            raise Exception("Source and target coordinate dimension not match")

    dimension = source_coord.shape[1]

    if metric == "great-circle":
        if dimension != 2:
            raise Exception("Great-circle distance only applicable to 2D coordinates")
        return np.array(
            [great_circle_distance(coord, target_coord) for coord in source_coord]
        ).astype(np.float32)

    if target_coord is None:
        return pdist(source_coord.astype(np.float32), metric, **kwargs).astype(
            np.float32
        )
    else:
        return cdist(
            source_coord.astype(np.float32),
            target_coord.astype(np.float32),
            metric,
            **kwargs,
        ).astype(np.float32)


@njit
def great_circle_distance(one_lonlat, many_lonlat):
    """
    Compute great-circle distance using Haversine algorithm.
    """

    lon_diff = np.radians(many_lonlat[:, 0] - one_lonlat[0])
    lat_diff = np.radians(many_lonlat[:, 1] - one_lonlat[1])
    lat_one = np.radians(one_lonlat[1])
    lat_many = np.radians(many_lonlat[:, 1])

    a = (
        np.sin(lat_diff / 2) ** 2
        + np.cos(lat_many) * np.cos(lat_one) * np.sin(lon_diff / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371.0

    return R * c
