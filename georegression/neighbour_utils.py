import numpy as np
from numba import njit, prange
from numpy import ndarray
from scipy.sparse import csr_array


def second_order_neighbour(neighbour_matrix, neighbour_leave_out=None):
    """
    Calculate second-order neighbour matrix.
    Args:
        neighbour_matrix: First-order neighbour matrix.
        neighbour_leave_out: The subset of neighbours that should be considered as first-order neighbour. If None, use neighbour_matrix.


    Returns:

    """
    if neighbour_leave_out is None:
        neighbour_leave_out = neighbour_matrix

    if isinstance(neighbour_matrix, ndarray):
        return _second_order_neighbour_dense(neighbour_matrix, neighbour_leave_out)
    elif isinstance(neighbour_matrix, csr_array):
        indices_list = _second_order_neighbour_sparse(
            neighbour_matrix.indptr,
            neighbour_matrix.indices,
            neighbour_leave_out.indptr,
            neighbour_leave_out.indices,
        )

        # Generate the indptr and indices for the sparse matrix.
        indptr = np.zeros((len(indices_list) + 1,), dtype=np.int32)
        for i in range(len(indices_list)):
            indptr[i + 1] = indptr[i] + len(indices_list[i])

        indices = np.hstack(indices_list)

        return csr_array((np.ones_like(indices), indices, indptr))

    raise ValueError("neighbour_matrix should be np.ndarray or csr_array.")


@njit(parallel=True)
def _second_order_neighbour_sparse(
    indptr, indices, indptr_leave_out, indices_leave_out
):
    N = len(indptr) - 1
    # Manually create the list with specified length to avoid parallel Mutating error.
    indices_list = [np.empty(0, dtype=np.int64)] * N
    for row_index in prange(N):
        neighbour_indices = indices_leave_out[
            indptr_leave_out[row_index] : indptr_leave_out[row_index + 1]
        ]
        second_neighbour_indices_union = np.zeros((N,))
        for neighbour_index in neighbour_indices:
            second_neighbour_indices = indices[
                indptr[neighbour_index] : indptr[neighbour_index + 1]
            ]
            for second_neighbour_index in second_neighbour_indices:
                second_neighbour_indices_union[second_neighbour_index] = True

        second_neighbour_indices_union = np.nonzero(second_neighbour_indices_union)[0]
        indices_list[row_index] = second_neighbour_indices_union

    return indices_list


@njit(parallel=True)
def _second_order_neighbour_dense(neighbour_matrix, neighbour_leave_out):
    second_order_matrix = np.empty_like(neighbour_matrix)
    for i in prange(neighbour_matrix.shape[0]):
        second_order_matrix[i] = np.sum(
            neighbour_matrix[neighbour_leave_out[i]], axis=0
        )
    return second_order_matrix


def neighbour_shrink(weight_matrix, shrink_rate=0.5):
    if isinstance(weight_matrix, np.ndarray):
        return _neighbour_shrink(weight_matrix, shrink_rate)
    elif isinstance(weight_matrix, csr_array):
        weight_matrix.data = _neighbour_shrink_sparse(weight_matrix.data, weight_matrix.indptr, shrink_rate)
        weight_matrix.eliminate_zeros()
        return weight_matrix > 0

@njit(parallel=True)
def _neighbour_shrink(weight_matrix: np.ndarray, shrink_rate=0.5):
    for i in prange(weight_matrix.shape[0]):
        neighbour_indices = np.nonzero(weight_matrix[i])[0]
        positive_value = weight_matrix[i]
        positive_value = positive_value[neighbour_indices]
        shrink_value = np.quantile(positive_value, shrink_rate)
        positive_value[positive_value < shrink_value] = 0
        # TODO: Rename j
        for j in range(len(neighbour_indices)):
            weight_matrix[i, neighbour_indices[j]] = positive_value[j]
    return weight_matrix > 0

@njit(parallel=True)
def _neighbour_shrink_sparse(weight_matrix_data, weight_matrix_indptr, shrink_rate=0.5):
    for i in prange(len(weight_matrix_indptr) - 1):
        positive_value = weight_matrix_data[
            weight_matrix_indptr[i] : weight_matrix_indptr[i + 1]
        ]
        shrink_value = np.quantile(positive_value, shrink_rate)
        positive_value[positive_value < shrink_value] = 0
        weight_matrix_data[
            weight_matrix_indptr[i] : weight_matrix_indptr[i + 1]
        ] = positive_value
    return weight_matrix_data


def sample_neighbour(weight_matrix, sample_rate=0.5, shrink_rate=0.5):
    """
    # TODO: More detailed description.

    Sample neighbour from weight matrix.
    Only the sampled neighbour will be used to fit the meta model.
    Therefore, the meta model will not be used for the sampled neighbour, but the out-of-sample neighbour.
    Args:
        weight_matrix:
        sample_rate:

    Returns:

    """

    # Do the shrink first.
    if shrink_rate is not None:
        neighbour_matrix = neighbour_shrink(weight_matrix, shrink_rate)
    else:
        neighbour_matrix = weight_matrix > 0

    # Do not sample itself.
    if isinstance(neighbour_matrix, np.ndarray):
        np.fill_diagonal(neighbour_matrix, False)
    elif isinstance(neighbour_matrix, csr_array):
        neighbour_matrix.setdiag(False)
        neighbour_matrix.eliminate_zeros()
    else:
        raise ValueError("weight_matrix should be np.ndarray or csr_array.")

    # Get the count to sample for each row.
    neighbour_count = np.sum(neighbour_matrix, axis=1)
    neighbour_count_sampled = np.ceil(neighbour_count * sample_rate).astype(int)
    neighbour_count_sampled[neighbour_count_sampled == 0] = 1
    neighbour_count_sampled[
        neighbour_count_sampled > neighbour_count
    ] = neighbour_count[neighbour_count_sampled > neighbour_count]

    neighbour_matrix_sampled = np.zeros(neighbour_matrix.shape, dtype=bool)

    # Set fixed random seed.
    np.random.seed(0)

    if isinstance(neighbour_matrix, np.ndarray):
        for i in range(neighbour_matrix.shape[0]):
            neighbour_matrix_sampled[
                i,
                np.random.choice(
                    # nonzero [0] for 1d array; [1] for 2d array.
                    np.nonzero(neighbour_matrix[i])[0],
                    neighbour_count_sampled[i],
                    replace=False,
                ),
            ] = True
    else:
        indices_list = []
        for i in range(neighbour_matrix.shape[0]):
            indices_list.append(
                # Sort the indices to make sure the structure of sparse matrix is correct.
                # But, really need to sort?
                np.sort(
                    # Leave out itself.
                    np.append(
                        np.random.choice(
                            neighbour_matrix.indices[
                                neighbour_matrix.indptr[i] : neighbour_matrix.indptr[
                                    i + 1
                                ]
                            ],
                            neighbour_count_sampled[i],
                            replace=False,
                        ),
                        i,
                    )
                )
            )

        indptr = np.zeros((len(indices_list) + 1,), dtype=np.int32)
        for i in range(len(indices_list)):
            indptr[i + 1] = indptr[i] + len(indices_list[i])

        indices = np.hstack(indices_list)
        neighbour_matrix_sampled = csr_array(
            (np.ones_like(indices), indices, indptr), dtype=bool
        )

    # Leave out itself.
    if isinstance(neighbour_matrix_sampled, np.ndarray):
        np.fill_diagonal(neighbour_matrix_sampled, True)

    return neighbour_matrix_sampled
