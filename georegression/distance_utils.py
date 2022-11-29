# TODO: https://jaykmody.com/blog/distance-matrices-with-numpy/
# TODO: https://stackoverflow.com/questions/22720864/efficiently-calculating-a-euclidean-distance-matrix-using-numpy

import numpy as np
from numba import njit


@njit
def great_circle_distance(one_lonlat, many_lonlat):
    """
    Compute great-circle distance using Haversine algorithm.
    """

    lon_diff = np.radians(many_lonlat[:, 0] - one_lonlat[0])
    lat_diff = np.radians(many_lonlat[:, 1] - one_lonlat[1])
    lat_one = np.radians(one_lonlat[1])
    lat_many = np.radians(many_lonlat[:, 1])

    a = np.sin(lat_diff / 2) ** 2 + np.cos(lat_many) * np.cos(lat_one) * np.sin(lon_diff / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371.0

    return R * c


def calculate_distance_one_to_many(one_coordinate_vector, many_coordinate_vector, distance_measure, p=None):
    """

    Args:
        one_coordinate_vector: shape of (1, Dimension of the coordinate)
        many_coordinate_vector: shape of (N, Dimension of the coordinate)
        distance_measure:
        p:

    Returns: shape of (N)

    """

    one_coordinate_vector = np.reshape(one_coordinate_vector, (1, -1))

    # Check the dimension of the vector/coordinate
    if one_coordinate_vector.shape[1] != many_coordinate_vector.shape[1]:
        raise Exception('Coordinate dimension not match')
    dimension = one_coordinate_vector.shape[1]

    coordinate_diff = many_coordinate_vector - one_coordinate_vector

    if distance_measure == 'euclidean':
        return np.sqrt(np.sum(coordinate_diff ** 2, axis=1))
    elif distance_measure == 'manhattan':
        return np.sum(np.abs(coordinate_diff), axis=1)
    elif distance_measure == 'minkowski':
        if p is None:
            raise Exception('Invalid p for minkowski')
        return np.sum(np.abs(coordinate_diff) ** p, axis=1) ** (1 / p)
    elif distance_measure == 'great-circle':
        if dimension != 2:
            raise Exception('Dimension must be 2 for longitude and latitude')
        return great_circle_distance(one_coordinate_vector, many_coordinate_vector)
    else:
        raise Exception('Unsupported distance measure')


def euclidean_distance_matrix(X, Y):
    """
    Implement from https://jaykmody.com/blog/distance-matrices-with-numpy/
    But it's not as efficient as said in the blog.
    Nevertheless, it's a good chance to dive deep into some calculation.
    """
    # this has the same affect as taking the dot product of each row with itself
    x2 = np.sum(X ** 2, axis=1)  # shape of (m)
    y2 = np.sum(Y ** 2, axis=1)  # shape of (n)

    # we can compute all x_i * y_j and store it in a matrix at xy[i][j] by
    # taking the matrix multiplication between X and X_train transpose
    # if you're stuggling to understand this, draw out the matrices and
    # do the matrix multiplication by hand
    # (m, d) x (d, n) -> (m, n)
    xy = np.matmul(X, Y.T)

    # each row in xy needs to be added with x2[i]
    # each column of xy needs to be added with y2[j]
    # to get everything to play well, we'll need to reshape
    # x2 from (m) -> (m, 1), numpy will handle the rest of the broadcasting for us
    # see: https://numpy.org/doc/stable/user/basics.broadcasting.html
    x2 = x2.reshape(-1, 1)
    dists = x2 - 2 * xy + y2  # (m, 1) repeat columnwise + (m, n) + (n) repeat rowwise -> (m, n)

    return dists
