from time import time

import dask
import dask.array as da
import dask_distance
import numpy as np
from dask.distributed import Client, LocalCluster
from dask.graph_manipulation import wait_on
from distributed import get_task_stream
from scipy import sparse

from georegression.weight_matrix import compound_weight


def test_dask_distance_matrix():
    count = 100000

    a = dask_distance.cdist(
        da.from_array(np.random.random((count, 2)), chunks=(2500, 2)),
        da.from_array(np.random.random((count, 2)), chunks=(2500, 2)),
        "euclidean",
    )

    a_sum = a.sum(axis=1)

    t_start = time()

    print(a_sum.compute())

    print(time() - t_start)


def test_dask_compatiblity():
    # 86 seconds for count=50000
    # 680 seconds for count=100000
    count = 100000
    distance_matrix = dask_distance.cdist(
        da.from_array(np.random.random((count, 2)), chunks={0: 4000, 1: 2}),
        da.from_array(np.random.random((count, 2)), chunks={0: 4000, 1: 2}),
        "euclidean",
    )
    distance_matrix = distance_matrix.rechunk({0: "auto", 1: -1})
    distance_matrix = wait_on(distance_matrix)

    # print(distance_matrix.mean().compute())
    # return

    t1 = time()
    result = compound_weight([distance_matrix], "bisquare", neighbour_count=0.1)
    t2 = time()
    print(t2 - t1)

    result_sparse = result.map_blocks(sparse.coo_matrix)

    t3 = time()
    print(result_sparse.compute())
    t4 = time()
    print(t4 - t3)


def test_dask_map_block():
    count = 10000
    distance_matrix = dask_distance.cdist(
        da.from_array(np.random.random((count, 2)), chunks={0: 4000, 1: 2}),
        da.from_array(np.random.random((count, 2)), chunks={0: 4000, 1: 2}),
        "euclidean",
    )
    # distance_matrix = distance_matrix.rechunk({0: 'auto', 1: -1})
    distance_matrix = wait_on(distance_matrix)

    t1 = time()

    percentile = distance_matrix.map_blocks(
        np.percentile,
        50,
        axis=1,
        keepdims=False,
        drop_axis=[1],
        # chunks=(distance_matrix.chunksize[0]),
    )

    print(percentile.shape, percentile.compute())
    t2 = time()
    print(t2 - t1)


def test_dask_reduction():
    count = 100000
    distance_matrix = dask_distance.cdist(
        da.from_array(np.random.random((count, 2)), chunks={0: 4000, 1: 2}),
        da.from_array(np.random.random((count, 2)), chunks={0: 4000, 1: 2}),
        "euclidean",
    )

    # 57.298909187316895 for rechunk
    # 40.120853900909424 for no rechunk
    # distance_matrix = distance_matrix.rechunk({0: 'auto', 1: -1})
    distance_matrix = wait_on(distance_matrix)

    t1 = time()

    def chunk_function(x, axis, keepdims):
        """
        Do the identical operation on a chunk of the data to pass to the aggregate function.
        Args:
            x (da.Array):
            axis:
            keepdims:

        Returns:

        """
        return x

    def aggregate_function(x, axis, keepdims):
        """
        Do the percentile operation on the aggregated (actually identity) data.

        Args:
            x (da.Array):
            axis:
            keepdims:

        Returns:

        """

        # Pre-call for dimensional checking by dask.
        if x.shape == (0, 0):
            return np.array([])
        return np.percentile(x, 99, axis=axis, keepdims=keepdims)

    percentile = da.reduction(
        distance_matrix,
        chunk_function,
        aggregate_function,
        axis=1,
        dtype=np.float64,
    )

    print(percentile.shape, percentile.compute())
    t2 = time()
    print(t2 - t1)


if __name__ == "__main__":
    # Set config of "distributed.comm.retry.count"
    dask.config.set({"distributed.comm.retry.count": 10})
    dask.config.set({"distributed.comm.timeouts.connect": 30})

    dask.config.get("distributed.worker.memory.target")
    dask.config.get("distributed.worker.memory.spill")
    dask.config.get("distributed.worker.memory.pause")
    dask.config.get("distributed.worker.memory.max-spill")
    # dask.config.set({"distributed.worker.memory.pause": 0.5})
    dask.config.set({"distributed.worker.memory.terminate": False})

    # create local cluster and start distributed scheduler.
    cluster = LocalCluster(
        local_directory="F:/dask",
        n_workers=4,
        memory_limit="6GiB",
    )
    client = Client(cluster)
    print(client.dashboard_link)

    with get_task_stream(plot="save", filename="task-stream.html") as ts:
        # test_dask_distance_matrix()
        test_dask_compatiblity()
        # test_dask_map_block()
        # test_dask_reduction()

    client.profile(filename="dask-profile.html")
