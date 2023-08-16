import dask.array as da
import dask_distance
import numpy as np
from dask.distributed import LocalCluster


def test_dask_distance_matrix():
    count = 100000

    a = dask_distance.cdist(
        da.from_array(np.random.random((count, 2)), chunks=(5000, 2)),
        da.from_array(np.random.random((count, 2)), chunks=(1000, 2)),
        "euclidean",
    )

    a.mean(axis=1).compute()

    step = int(count / 10)
    for i in range(10):
        print(a[i * step : (i + 1) * step].compute())


if __name__ == "__main__":
    cluster = LocalCluster(n_workers=16)
    print(cluster)

if __name__ == "__main__":
    test_dask_distance_matrix()
