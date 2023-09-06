import numpy as np
from scipy.sparse import csr_matrix


def test_sparse_operation():
    arr = (np.random.random((10, 10)) - 0.8) > 0
    # TODO: Consider using lil_array for modifying the array.
    arr = csr_matrix(arr, dtype=bool)

    # Analogy of nonzere. Ref:https://numpy.org/devdocs/user/basics.indexing.html#boolean-array-indexing
    second_nei = np.sum(arr[arr[0].nonzero()[1]], axis=0) > 0
    print(second_nei)

    # Set value
    second_neighbour_matrix = csr_matrix((10, 10), dtype=bool)
    second_neighbour_matrix[0, second_nei.nonzero()[1]] = True
    print(second_neighbour_matrix)

    # Index array by sparse matrix
    np.random.random((10, 10))[second_neighbour_matrix[0].nonzero()]