# TODO: https://jaykmody.com/blog/distance-matrices-with-numpy/
# TODO: https://stackoverflow.com/questions/22720864/efficiently-calculating-a-euclidean-distance-matrix-using-numpy
# TODO: Ref to https://github.com/talboger/fastdist and https://github.com/numba/numba-scipy/issues/38#issuecomment-623569703 to speed up by parallel computing
import dask
import dask.array as da
import dask_distance
import numpy as np
from distributed import LocalCluster, Client
from numba import njit
from scipy.spatial.distance import pdist, cdist
from os.path import join
from pathlib import Path

def distance_matrices(
    source_coords: list[np.ndarray], target_coords=None, metrics='euclidean',
        use_dask=False, cache_sort=False, args=None, **kwargs
):
    """
    Check the validatiton of the parameter list.
    If single value is provided, convert it to a list with length equal to the dimension of the vector list.
    Then, call the distance_matrix function for each dimension.

    Args:
        source_coords:
        target_coords:
        metrics:
        use_dask:
        args:

    Returns:

    """

    dimension = len(source_coords)

    # Check equal length of source and target coordinates
    if target_coords is not None:
        if dimension != len(target_coords):
            raise Exception("Source and target coordinate length not match")
    else:
        target_coords = [None] * dimension

    # Check whether the input parameters are lists, and if not, convert them to lists with length equal to the dimension of the vector list.
    if not isinstance(metrics, list):
        metrics = [metrics] * dimension

    if args is None:
        args = [kwargs] * dimension

    return [
        distance_matrix(
            source_coords[dim],
            target_coords[dim],
            metrics[dim],
            use_dask,
            cache_sort,
            **args[dim],
        )
        for dim in range(dimension)
    ]


def distance_matrix(source_coord, target_coord, metric, use_dask, cache_sort, **kwargs):
    # Check equal dimension of source and target coordinates
    if target_coord is not None:
        if source_coord.shape[1] != target_coord.shape[1]:
            raise Exception("Source and target coordinate dimension not match")

    dimension = source_coord.shape[1]

    if use_dask:
        filepath = kwargs.get("filepath", "distance_matrix.zarr")

        if not filepath.endswith(".zarr"):
            filepath = filepath + ".zarr"

        if (kwargs.get("filepath", None) is not None) and (not kwargs.get("overwrite", False)):
            if Path(filepath).exists():
                return da.from_zarr(filepath)

        if target_coord is None:
            distance_matrix = dask_distance.pdist(source_coord, metric=metric)
        else:
            distance_matrix = dask_distance.cdist(source_coord, target_coord, metric=metric)

        distance_matrix = distance_matrix.rechunk({0: "auto", 1: -1})
        distance_matrix.to_zarr(filepath, overwrite=True)

        if cache_sort:
            distance_matrix_sorted = distance_matrix.map_blocks(np.sort)
            distance_matrix_sorted.to_zarr(filepath.replace(".zarr", "_sorted.zarr"), overwrite=True)

            return distance_matrix, distance_matrix_sorted

        return distance_matrix

    else:
        if metric == "great-circle":
            if dimension != 2:
                raise Exception("Great-circle distance only applicable to 2D coordinates")
            return np.array(
                [great_circle_distance(coord, target_coord) for coord in source_coord]
            ).astype(np.float32)

        if target_coord is None:
            return pdist(source_coord.astype(np.float32), metric, **kwargs).astype(np.float32)
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
