from georegression.distance_utils import (
    euclidean_distance_matrix,
    calculate_distance_one_to_many,
)
import numpy as np
from time import time
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import distance_matrix
from sklearn.metrics import pairwise_distances

X = np.random.random((30000, 2))


def one_loop_version(X, Y):
    m = X.shape[0]
    n = Y.shape[0]
    dist = np.empty((m, n))
    for i in range(m):
        dist[i, :] = np.sqrt(np.sum((X[i] - Y) ** 2, axis=1))


def test_distance_matrix():
    t1 = time()
    euclidean_distance_matrix(X, X)
    t2 = time()
    for x in X:
        calculate_distance_one_to_many(x, X, "euclidean")
    t3 = time()
    pdist(X)
    t4 = time()
    cdist(X, X)
    t5 = time()
    one_loop_version(X, X)
    t6 = time()
    distance_matrix(X, X)
    t7 = time()
    pairwise_distances(X, X)
    t8 = time()
    pairwise_distances(X, X, n_jobs=-1)
    t9 = time()

    print(t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6, t8 - t7, t9 - t8)
    # For (30000,2):
    # INTEL I7 8700, PYTHON 3.7
    # 65.28857350349426 12.295624017715454 2.637856960296631 4.8066534996032715 14.165263652801514 20.550457000732422
    # AMD R9 7950X, PYTHON 3.11
    # 8.726486682891846 9.398136377334595 1.1142053604125977 2.3201327323913574 10.608065605163574 13.829639911651611 5.806329011917114


def matrix_size(dtype=None):
    # Record the time elapsed for creating the distance matrix.
    t1 = time()
    X = np.random.random((30000, 30000)).astype(dtype)
    t2 = time()
    print(t2 - t1)

    # Calculate the memory usage of the matrix. Print in MB.
    print(X.nbytes / 1024**2, "MB")

    # Calculate the time elapsed for calculating the distance matrix.
    X = np.random.random((30000, 2)).astype(dtype)
    t1 = time()
    pdist(X)
    t2 = time()
    pairwise_distances(X, X, n_jobs=-1)
    t3 = time()
    print(t2 - t1, t3 - t2)


if __name__ == "__main__":
    # test_distance_matrix()
    matrix_size(np.float16)
