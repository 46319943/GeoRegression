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
    source_coords: list[np.ndarray], target_coords=None, metrics=None, use_dask=False, args=None
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
            use_dask,
            **args[dim],
        )
        for dim in range(dimension)
    ]


def distance_matrix(source_coord, target_coord=None, metric=None, use_dask=False, **kwargs):
    # Check equal dimension of source and target coordinates
    if target_coord is not None:
        if source_coord.shape[1] != target_coord.shape[1]:
            raise Exception("Source and target coordinate dimension not match")

    dimension = source_coord.shape[1]

    if use_dask:
        filepath = join(
               kwargs.get("local_directory", ''),
               kwargs.get("filename", "distance_matrix.zarr")
           )

        if (kwargs.get("filename", None) is not None) and (not kwargs.get("overwrite", False)):
            # Whether file exists
            if Path(filepath).exists():
                return da.from_zarr(filepath)

        dask.config.set({"distributed.comm.retry.count": 10})
        dask.config.set({"distributed.comm.timeouts.connect": 30})
        dask.config.set({"distributed.worker.memory.terminate": False})

        cluster = LocalCluster(local_directory=kwargs.get("local_directory", None))
        client = Client(cluster)
        print(client.dashboard_link)

        distance_matrix = dask_distance.cdist(source_coord, source_coord, metric="euclidean")
        distance_matrix = distance_matrix.rechunk({0: "auto", 1: -1})
        distance_matrix = distance_matrix.map_blocks(np.sort)

        distance_matrix.to_zarr(filepath, overwrite=True)
        return distance_matrix


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
