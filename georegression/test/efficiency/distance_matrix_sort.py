import dask
import dask.array as da
from distributed import LocalCluster, Client
import numpy as np

from georegression.distance_utils import _distance_matrix, _distance_matrices
from georegression.kernel import adaptive_bandwidth
from georegression.weight_matrix import weight_matrix_from_distance
from scipy import sparse




def test_distance_matrix_using_dask():
    dask.config.set({"distributed.comm.retry.count": 10})
    dask.config.set({"distributed.comm.timeouts.connect": 30})
    dask.config.set({"distributed.worker.memory.terminate": False})
    
    cluster = LocalCluster(local_directory="F://dask")
    client = Client(cluster)
    print(client.dashboard_link)

    count = 50000

    _distance_matrices(
        [da.from_array(np.random.random((count, 2)), chunks=(4000, 2))],
        [da.from_array(np.random.random((count, 2)), chunks=(4000, 2))],
        use_dask=True,
        cache_sort=True,
        filepath="F://test_distance_matrix",
        overwrite=True,
    )


def test_weight_matrix_using_sorted_distance_matrix():
    dask.config.set({"distributed.comm.retry.count": 10})
    dask.config.set({"distributed.comm.timeouts.connect": 30})
    dask.config.set({"distributed.worker.memory.terminate": False})
    
    cluster = LocalCluster(local_directory="F://dask")
    client = Client(cluster)
    print(client.dashboard_link)

    distance_matrix = da.from_zarr("F://dask//test_distance_matrix.zarr")
    distance_matrix_sorted = da.from_zarr("F://dask//test_distance_matrix_sorted.zarr")

    # bandwidth = adaptive_bandwidth(distance_matrix_sorted, 2)
    # print(bandwidth.compute())

    weight_matrix = weight_matrix_from_distance(
        [distance_matrix],
        "bisquare", neighbour_count=0.01,
        distance_matrices_sorted=[distance_matrix_sorted]
    )
    weight_matrix_sparse = weight_matrix.map_blocks(sparse.coo_matrix)
    print(weight_matrix_sparse.compute())


def test_dask_client():
    dask.config.set({"distributed.comm.retry.count": 10})
    dask.config.set({"distributed.comm.timeouts.connect": 30})
    dask.config.set({"distributed.worker.memory.terminate": False})
    
    cluster = LocalCluster(local_directory=kwargs.get("local_directory", None))
    client = Client(cluster)
    print(client.dashboard_link)

    Client.get()


if __name__ == "__main__":
    # test_distance_matrix_using_dask()
    test_weight_matrix_using_sorted_distance_matrix()

    pass
