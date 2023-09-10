import dask.array as da
import numpy as np

from georegression.distance_utils import distance_matrix
from georegression.kernel import adaptive_bandwidth


def test_distance_matrix_using_dask():
    count = 1000

    distance_matrix(
        da.from_array(np.random.random((count, 2)), chunks=(250, 2)),
        da.from_array(np.random.random((count, 2)), chunks=(250, 2)),
        use_dask=True,
        local_directory="F://dask",
        filename="test_distance_matrix",
    )


def test_weight_matrix_using_sorted_distance_matrix():
    distance_matrix_sorted = da.from_zarr("F://dask//test_distance_matrix")
    bandwidth = adaptive_bandwidth(distance_matrix_sorted, 2)
    print(bandwidth.compute())


if __name__ == "__main__":
    # test_distance_matrix_using_dask()
    test_weight_matrix_using_sorted_distance_matrix()

    pass
