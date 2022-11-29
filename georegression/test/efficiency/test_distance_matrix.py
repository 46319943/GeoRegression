from georegression.distance_utils import euclidean_distance_matrix, calculate_distance_one_to_many
import numpy as np
from time import time
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import distance_matrix

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
        calculate_distance_one_to_many(x, X, 'euclidean')
    t3 = time()
    pdist(X)
    t4 = time()
    cdist(X, X)
    t5 = time()
    one_loop_version(X, X)
    t6 = time()
    distance_matrix(X, X)
    t7 = time()

    print(t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6)
    # For (30000,2):
    # 65.28857350349426 12.295624017715454 2.637856960296631 4.8066534996032715 14.165263652801514 20.550457000732422


if __name__ == '__main__':
    test_distance_matrix()
