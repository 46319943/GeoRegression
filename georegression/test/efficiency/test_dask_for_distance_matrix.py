from time import time

import dask.array as da
import dask_distance
import numpy as np
from dask.distributed import Client, LocalCluster
from distributed import get_task_stream

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

    with get_task_stream(plot='save', filename='task-stream.html') as ts:
        print(a_sum.compute())

    client.profile(filename="dask-profile.html")

    print(time() - t_start)


def test_dask_compatiblity():
    distance_matrix = da.random.random((1000, 1000), chunks=(100, 100))
    result = compound_weight([distance_matrix], "bisquare", neighbour_count=0.1)

    print(result.compute())


if __name__ == "__main__":
    # create local cluster and start distributed scheduler.
    cluster = LocalCluster()
    client = Client(cluster)
    print(client.dashboard_link)

if __name__ == "__main__":
    # test_dask_distance_matrix()
    test_dask_compatiblity()
