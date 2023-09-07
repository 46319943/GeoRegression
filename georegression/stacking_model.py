import math
from itertools import compress
from time import time

import numpy as np
from numba import njit, prange
from scipy.sparse import csr_array
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from slab_utils.quick_logger import logger

from georegression.numba_impl import ridge_cholesky
from georegression.weight_matrix import calculate_compound_weight_matrix
from georegression.weight_model import WeightModel


def second_order_neighbour(neighbour_matrix, neighbour_leave_out=None):
    """
    Calculate second-order neighbour matrix.
    Args:
        neighbour_matrix: First-order neighbour matrix

    Returns:

    """

    # TODO: Use parallel or sparse matrix to speed up.

    if neighbour_leave_out is None:
        neighbour_leave_out = neighbour_matrix

    if isinstance(neighbour_matrix, np.ndarray):
        return _second_order_neighbour(neighbour_matrix, neighbour_leave_out)
    else:
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


@njit()
def _second_order_neighbour_sparse(
    indptr, indices, indptr_leave_out, indices_leave_out
):
    N = len(indptr) - 1
    indices_list = []
    for row_index in range(N):
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
        indices_list.append(second_neighbour_indices_union)

    return indices_list


def _second_order_neighbour(neighbour_matrix, neighbour_leave_out):
    second_order_matrix = np.empty_like(neighbour_matrix)
    for i in prange(neighbour_matrix.shape[0]):
        second_order_matrix[i] = np.sum(
            neighbour_matrix[neighbour_leave_out[i]], axis=0
        )
    return second_order_matrix


def sample_neighbour(weight_matrix, sample_rate=0.5):
    """
    Sample neighbour from weight matrix.
    Only the sampled neighbour will be used to fit the meta model.
    Therefore, the meta model will not be used for the sampled neighbour, but the out-of-sample neighbour.
    Args:
        weight_matrix:
        sample_rate:

    Returns:

    """

    neighbour_matrix = weight_matrix > 0

    # Do not sample itself.
    if isinstance(neighbour_matrix, np.ndarray):
        np.fill_diagonal(neighbour_matrix, False)
    else:
        neighbour_matrix.setdiag(False)
        neighbour_matrix.eliminate_zeros()

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
        # TODO: Optimize for sparse matrix.
        indices_list = []
        for i in range(neighbour_matrix.shape[0]):
            indices_list.append(
                # Sort the indices to make sure the structure of sparse matrix is correct.
                # But, really need to sort?
                np.sort(
                    # Leave out itself.
                    np.append(
                        np.random.choice(
                            # nonzero [0] for 1d array; [1] for 2d array.
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


class StackingWeightModel(WeightModel):
    def __init__(
        self,
        local_estimator,
        # Weight matrix param
        distance_measure=None,
        kernel_type=None,
        distance_ratio=None,
        bandwidth=None,
        neighbour_count=None,
        midpoint=None,
        distance_args=None,
        # Model param
        leave_local_out=True,
        sample_local_rate=None,
        cache_data=False,
        cache_estimator=False,
        n_jobs=-1,
        alpha=10.0,
        neighbour_leave_out_rate=None,
        estimator_sample_rate=None,
        use_numba=True,
        *args,
        **kwargs
    ):
        super().__init__(
            local_estimator,
            distance_measure=distance_measure,
            kernel_type=kernel_type,
            distance_ratio=distance_ratio,
            bandwidth=bandwidth,
            neighbour_count=neighbour_count,
            midpoint=midpoint,
            distance_args=distance_args,
            # Model param
            leave_local_out=leave_local_out,
            sample_local_rate=sample_local_rate,
            cache_data=cache_data,
            cache_estimator=cache_estimator,
            n_jobs=n_jobs,
            *args,
            **kwargs
        )

        self.meta_estimator_list = None
        self.stacking_predict_ = None
        self.llocv_stacking_ = None
        self.alpha = alpha
        self.neighbour_leave_out_rate = neighbour_leave_out_rate
        self.estimator_sample_rate = estimator_sample_rate
        self.use_numba = use_numba

    def fit(self, X, y, coordinate_vector_list=None, weight_matrix=None):
        """
        Fit an estimator at every location using the local data.
        Then, given a location, use the neighbour estimators to blending a new estimator, fitted also by the local data.

        Args:
            X:
            y:
            coordinate_vector_list ():
            weight_matrix:

        Returns:

        """

        cache_estimator = self.cache_estimator
        self.cache_estimator = True

        if weight_matrix is None:
            weight_matrix = calculate_compound_weight_matrix(
                coordinate_vector_list,
                coordinate_vector_list,
                self.distance_measure,
                self.kernel_type,
                self.distance_ratio,
                self.bandwidth,
                self.neighbour_count,
                self.distance_args,
            )

        t_neighbour_start = time()

        # Do the leave out neighbour sampling.
        neighbour_leave_out = None
        if self.neighbour_leave_out_rate is not None:
            neighbour_leave_out = sample_neighbour(
                weight_matrix, self.neighbour_leave_out_rate
            )

            if not isinstance(neighbour_leave_out, np.ndarray):
                neighbour_leave_out_ = neighbour_leave_out

            # From (i,j) is that i-th observation will not be used to fit the j-th base estimator
            # so that the j-th base estimator will be used for meta-estimator.
            # To (j,i) is that j-th observation will not consider i-th observation as neighbour while fitting base estimator.

            if isinstance(neighbour_leave_out, np.ndarray):
                neighbour_leave_out = neighbour_leave_out.T
            else:
                # Structure not change for sparse matrix. BUG HERE.
                neighbour_leave_out = csr_array(neighbour_leave_out.T)
            # weight_matrix = weight_matrix * ~neighbour_leave_out

        t_neighbour_end = time()
        logger.debug("End of Neighbour leave out")

        # TODO: Consider the phenomenon that weight_matrix_local[neighbour_leave_out.nonzero()] is not zero.

        # weight_matrix_local = weight_matrix.copy()
        # To set the value for sparse matrix, convert it first to lil_array, then convert back to csr_array.
        # This can make sure the inner structure of csr_array is correct to be able to manipulate directly .
        # weight_matrix_local = lil_array(weight_matrix_local)
        # weight_matrix_local[neighbour_leave_out.nonzero()] = 0
        # weight_matrix_local = csr_array(weight_matrix_local)

        weight_matrix_local = weight_matrix.copy()
        weight_matrix_local[neighbour_leave_out.nonzero()] = 0
        if isinstance(weight_matrix_local, csr_array):
            # Or just use eliminate_zeros() to remove the zero elements.
            weight_matrix_local.eliminate_zeros()

        super().fit(
            X,
            y,
            coordinate_vector_list=coordinate_vector_list,
            weight_matrix=weight_matrix_local,
        )

        neighbour_matrix = weight_matrix > 0

        # Indicator of input data for each local estimator.
        # Before the local itself is set False in neighbour_matrix. Avoid no meta prediction for local.
        t_second_order_start = time()
        second_neighbour_matrix = second_order_neighbour(
            neighbour_matrix, neighbour_leave_out
        )
        t_second_order_end = time()
        logger.debug("End of Second order neighbour matrix")

        if isinstance(neighbour_matrix, np.ndarray):
            np.fill_diagonal(neighbour_matrix, False)
        elif isinstance(neighbour_matrix, csr_array):
            # BUG HERE. setdiag doesn't change the structure (indptr, indices), only data change from True to False.
            neighbour_matrix.setdiag(False)
            # TO FIX: Just use eliminate_zeros
            neighbour_matrix.eliminate_zeros()

        self.cache_estimator = cache_estimator
        self.meta_estimator_list = self.local_estimator_list
        self.local_estimator_list = None



        # Iterate the stacking estimator list to get the transformed X meta.
        # Cache all the data that will be used by neighbour estimators in one iteration by using second_neighbour_matrix.
        # First dimension is data index, second dimension is estimator index.
        # X_meta[i, j] means the prediction of estimator j on data i.
        t_predict_s = time()

        if isinstance(second_neighbour_matrix, np.ndarray):
            X_meta = np.zeros((self.N, self.N))
            for i in range(self.N):
                if second_neighbour_matrix[i].any():
                    X_meta[second_neighbour_matrix[i], i] = self.meta_estimator_list[
                        i
                    ].predict(X[second_neighbour_matrix[i]])
        else:
            prediction_result = list()
            for i in range(self.N):
                if (
                    second_neighbour_matrix.indptr[i]
                    != second_neighbour_matrix.indptr[i + 1]
                ):
                    prediction_result.append(
                        self.meta_estimator_list[i].predict(
                            X[
                                second_neighbour_matrix.indices[
                                    second_neighbour_matrix.indptr[
                                        i
                                    ] : second_neighbour_matrix.indptr[i + 1]
                                ]
                            ]
                        )
                    )
            X_meta_T = csr_array(
                (
                    np.hstack(prediction_result),
                    second_neighbour_matrix.indices,
                    second_neighbour_matrix.indptr,
                )
            )
            X_meta = X_meta_T.getH().tocsr()

        t_predict_e = time()
        logger.debug("End of Meta estimator prediction")

        t_transpose_start = time()
        if isinstance(X_meta, np.ndarray):
            X_meta_T = X_meta.transpose().copy(order="C")

        t_transpost_end = time()
        logger.debug(
            "End of Transpose meta estimator prediction",
        )

        local_stacking_predict = []
        local_stacking_estimator_list = []
        indexing_time = 0
        stacking_time = 0

        if isinstance(neighbour_leave_out, np.ndarray):
            for i in range(self.N):
                # TODO: Use RidgeCV to find best alpha
                final_estimator = Ridge(alpha=self.alpha, solver="lsqr")

                # TODO: Consider whether to add the meta prediction of the local meta estimator.
                t_indexing_start = time()

                neighbour_sample = neighbour_matrix[[i], :]

                if self.neighbour_leave_out_rate is not None:
                    # neighbour_sample = neighbour_leave_out[i]
                    neighbour_sample = neighbour_leave_out[:, i]
                    # neighbour_sample = neighbour_leave_out_[[i]]

                # Sample from neighbour bool matrix to get sampled neighbour index.
                if self.estimator_sample_rate is not None:
                    neighbour_indexes = np.nonzero(neighbour_sample[i])

                    neighbour_indexes = np.random.choice(
                        neighbour_indexes[0],
                        math.ceil(
                            neighbour_indexes[0].shape[0] * self.estimator_sample_rate
                        ),
                        replace=False,
                    )
                    # Convert back to bool matrix.
                    neighbour_sample = np.zeros_like(neighbour_matrix[i])
                    neighbour_sample[neighbour_indexes] = 1

                X_fit = X_meta_T[neighbour_sample][:, neighbour_matrix[i]].T
                y_fit = y[neighbour_matrix[i]]
                t_indexing_end = time()

                t_stacking_start = time()
                final_estimator.fit(
                    X_fit, y_fit, sample_weight=weight_matrix[i, neighbour_matrix[i]]
                )
                t_stacking_end = time()

                local_stacking_predict.append(
                    final_estimator.predict(
                        np.expand_dims(X_meta[i, neighbour_sample], 0)
                    )
                )

                # TODO: Unordered coef for each estimator.
                stacking_estimator = StackingEstimator(
                    final_estimator,
                    list(compress(self.meta_estimator_list, neighbour_sample)),
                )
                local_stacking_estimator_list.append(stacking_estimator)

                indexing_time = indexing_time + t_indexing_end - t_indexing_start
                stacking_time = stacking_time + t_stacking_end - t_stacking_start

            self.stacking_predict_ = local_stacking_predict
            self.llocv_stacking_ = r2_score(self.y_sample_, local_stacking_predict)
            self.local_estimator_list = local_stacking_estimator_list

        elif isinstance(neighbour_leave_out, csr_array) and not self.use_numba:
            for i in range(self.N):
                # TODO: Use RidgeCV to find best alpha
                final_estimator = Ridge(alpha=self.alpha, solver='cholesky')

                # TODO: Consider whether to add the meta prediction of the local meta estimator.
                t_indexing_start = time()

                # neighbour_sample = neighbour_leave_out[:, [i]]
                # neighbour_sample = neighbour_leave_out_[[i]]

                # Wrong leave out neighbour cause partial data leak.
                # neighbour_leave_out_indices = neighbour_leave_out.indices[
                #                               neighbour_leave_out.indptr[i]:neighbour_leave_out.indptr[i + 1]
                #                               ]
                neighbour_leave_out_indices = neighbour_leave_out_.indices[
                    neighbour_leave_out_.indptr[i] : neighbour_leave_out_.indptr[i + 1]
                ]
                neighbour_indices = neighbour_matrix.indices[
                    neighbour_matrix.indptr[i] : neighbour_matrix.indptr[i + 1]
                ]

                X_fit = (
                    X_meta_T[neighbour_leave_out_indices][:, neighbour_indices].toarray().T
                )
                y_fit = y[neighbour_indices]
                t_indexing_end = time()

                t_stacking_start = time()
                final_estimator.fit(
                    X_fit, y_fit, sample_weight=weight_matrix[[i], neighbour_indices]
                )
                t_stacking_end = time()

                local_stacking_predict.append(
                    final_estimator.predict(
                        np.expand_dims(X_meta[[i], neighbour_leave_out_indices], 0)
                    )
                )

                # TODO: Unordered coef for each estimator.
                stacking_estimator = StackingEstimator(
                    final_estimator,
                    [
                        self.meta_estimator_list[leave_out_index]
                        for leave_out_index in neighbour_leave_out_indices
                    ],
                )
                local_stacking_estimator_list.append(stacking_estimator)

                indexing_time = indexing_time + t_indexing_end - t_indexing_start
                stacking_time = stacking_time + t_stacking_end - t_stacking_start

                # break

            self.stacking_predict_ = local_stacking_predict
            self.llocv_stacking_ = r2_score(self.y_sample_, local_stacking_predict)
            self.local_estimator_list = local_stacking_estimator_list

            print(self.llocv_stacking_)

        else:
            @njit(parallel=True)
            def stacking_numba(
                leave_out_matrix_indptr,
                leave_out_matrix_indices,
                neighbour_matrix_indptr,
                neighbour_matrix_indices,
                X_meta_T_indptr,
                X_meta_T_indices,
                X_meta_T_data,
                y,
                weight_matrix_indptr,
                weight_matrix_indices,
                weight_matrix_data,
                alpha,
            ):
                N = len(leave_out_matrix_indptr) - 1
                coef_list = [np.empty((0, 0))] * N
                intercept_list = [np.empty(0)] * N
                y_predict_list = [np.empty(0)] * N

                for i in prange(N):
                    leave_out_indices = leave_out_matrix_indices[
                        leave_out_matrix_indptr[i] : leave_out_matrix_indptr[i + 1]
                    ]
                    neighbour_indices = neighbour_matrix_indices[
                        neighbour_matrix_indptr[i] : neighbour_matrix_indptr[i + 1]
                    ]

                    # Find the index of the first element equals i
                    # for index_i in range(len(neighbour_indices)):
                    #     if neighbour_indices[index_i] == i:
                    #         break

                    # Delete self from neighbour_indices
                    # neighbour_indices = np.hstack((neighbour_indices[:index_i], neighbour_indices[index_i + 1:]))
                    neighbour_indices = neighbour_indices[neighbour_indices != i]

                    X_fit_T = np.zeros((len(leave_out_indices), len(neighbour_indices)))

                    # Needed to sort?
                    # leave_out_indices = np.sort(leave_out_indices)

                    for X_fit_row_index in range(len(leave_out_indices)):
                        neighbour_available_indices = X_meta_T_indices[
                            X_meta_T_indptr[
                                leave_out_indices[X_fit_row_index]
                            ] : X_meta_T_indptr[leave_out_indices[X_fit_row_index] + 1]
                        ]
                        current_column = 0
                        for available_iter_i in range(len(neighbour_available_indices)):
                            if (
                                neighbour_available_indices[available_iter_i]
                                in neighbour_indices
                            ):
                                X_fit_T[X_fit_row_index, current_column] = X_meta_T_data[
                                    X_meta_T_indptr[leave_out_indices[X_fit_row_index]]
                                    + available_iter_i
                                ]
                                current_column = current_column + 1

                    y_fit = y[neighbour_indices]

                    weight_indices = weight_matrix_indices[
                        weight_matrix_indptr[i] : weight_matrix_indptr[i + 1]
                    ]
                    # weight_indices = weight_indices[weight_indices != i]
                    weight_fit = weight_matrix_data[
                        weight_matrix_indptr[i] : weight_matrix_indptr[i + 1]
                    ]
                    weight_fit = weight_fit[weight_indices != i]

                    # weight_fit = np.hstack((weight_fit[:index_i], weight_fit[index_i + 1:]))

                    coef, intercept = ridge_cholesky(X_fit_T.T, y_fit, alpha, weight_fit)

                    X_predict = np.zeros((len(leave_out_indices),))
                    for X_predict_row_index in range(len(leave_out_indices)):
                        neighbour_available_indices = X_meta_T_indices[
                            X_meta_T_indptr[
                                leave_out_indices[X_predict_row_index]
                            ] : X_meta_T_indptr[leave_out_indices[X_predict_row_index] + 1]
                        ]

                        # Find the index of the first element equals i
                        has_found = False
                        for available_iter_i in range(len(neighbour_available_indices)):
                            if neighbour_available_indices[available_iter_i] == i:
                                has_found = True
                                break

                        if has_found == False:
                            print()

                        X_predict[X_predict_row_index] = X_meta_T_data[
                            X_meta_T_indptr[leave_out_indices[X_predict_row_index]]
                            + available_iter_i
                        ]

                    y_predict = np.dot(X_predict, coef) + intercept

                    coef_list[i] = coef.T
                    intercept_list[i] = intercept
                    y_predict_list[i] = y_predict

                return coef_list, intercept_list, y_predict_list

            t1 = time()
            # Different solver makes a little difference.
            coef_list, intercept_list, y_predict_list = stacking_numba(
                neighbour_leave_out_.indptr,
                neighbour_leave_out_.indices,
                neighbour_matrix.indptr,
                neighbour_matrix.indices,
                X_meta_T.indptr,
                X_meta_T.indices,
                X_meta_T.data,
                y,
                weight_matrix.indptr,
                weight_matrix.indices,
                weight_matrix.data,
                self.alpha,
            )
            t2 = time()
            logger.debug("Numba running time: %s \n", t2 - t1)

            self.stacking_predict_ = np.array(y_predict_list)
            self.llocv_stacking_ = r2_score(self.y_sample_, self.stacking_predict_)
            # TODO
            # self.local_estimator_list = local_stacking_estimator_list

            print(self.llocv_stacking_)

        # Log the time elapsed in a single line
        logger.debug(
            "Leave local out elapsed: %s \n"
            "Second order neighbour matrix elapsed: %s \n"
            "Meta estimator prediction elapsed: %s \n"
            "Transpose meta estimator prediction elapsed: %s \n"
            "Indexing time: %s \n"
            "Stacking time: %s \n",
            t_neighbour_end - t_neighbour_start,
            t_second_order_end - t_second_order_start,
            t_predict_e - t_predict_s,
            t_transpost_end - t_transpose_start,
            indexing_time,
            stacking_time,
        )

        return self

    # TODO: Implement predict_by_fit


class StackingEstimator(BaseEstimator):
    def __init__(self, final_estimator, meta_estimators):
        self.final_estimator = final_estimator
        self.meta_estimators = meta_estimators

    def predict(self, X):
        X_meta = [meta_estimator.predict(X) for meta_estimator in self.meta_estimators]
        X_meta = np.hstack(X_meta)
        return self.final_estimator.predict(X_meta)
