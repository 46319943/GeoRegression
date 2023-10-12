import math
from itertools import compress
from time import time

import numpy as np
from numba import njit, prange
from scipy.sparse import csr_array
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.utils import check_X_y
from slab_utils.quick_logger import logger
from joblib import Parallel, delayed

from georegression.neighbour_utils import (
    second_order_neighbour,
    sample_neighbour,
    neighbour_shrink,
)
from georegression.numba_impl import ridge_cholesky
from georegression.weight_matrix import weight_matrix_from_points
from georegression.weight_model import WeightModel
from georegression.numba_impl import r2_score as r2_numba


def _fit_local_estimator(
        local_estimator, X, y,
        sample_weight,
        X_second_neighbour,
        local_x, return_estimator=False
):
    """
    Wrapper for parallel fitting.
    """

    # TODO: Add partial calculation for non-cache solution.

    local_estimator.fit(X, y, sample_weight=sample_weight)
    local_predict = local_estimator.predict(local_x.reshape(1, -1))
    second_neighbour_predict = local_estimator.predict(X_second_neighbour)

    if return_estimator:
        return local_predict, second_neighbour_predict, local_estimator
    else:
        return local_predict, second_neighbour_predict, None

def _fit(
    X,
    y,
    estimator_list,
    weight_matrix,
    second_neighbour_matrix,
    local_indices=None,
    cache_estimator=False,
    X_predict=None,
    n_patches=None,
):
    """
    Using joblib to parallelize the meta predicting process fails to accelerate, because the overhead of pickling/unpickling the model is too heavy.
    This is a compromise solution to incorporate the second neighbour prediction procedure into the fitting process.
    Actually, no so much work to implement this than I assumed before.
    # TODO: To better solve the prorblem, using numba or cython or other language to fully utilize the multicore.

    """

    t_start = time()

    # Generate the mask of selection from weight matrix. Select non-zero weight to avoid zero weight input.
    neighbour_matrix = weight_matrix != 0

    # Use all data if sample indices not provided.
    N = weight_matrix.shape[0]
    if local_indices is None:
        local_indices = range(N)

    # Data used for local prediction. Different from X when source and target are not same for weight matrix.
    if X_predict is None:
        X_predict = X

    # Parallel run the job. return [(prediction, estimator), (), ...]
    if isinstance(weight_matrix, np.ndarray):
        def batch_wrapper(local_indices):
            local_prediction_list = []
            second_neighbour_prediction_list = []
            local_estimator_list = []
            for i in local_indices:
                estimator = estimator_list[i]
                neighbour_mask = neighbour_matrix[i]
                row_weight = weight_matrix[i]
                x = X_predict[i]
                local_predict, second_neighbour_predict, local_estimator = _fit_local_estimator(
                    estimator, X[neighbour_mask], y[neighbour_mask], local_x=x,
                    sample_weight=row_weight[neighbour_mask], X_second_neighbour=X[second_neighbour_matrix[i]],
                    return_estimator=cache_estimator
                )
                local_prediction_list.append(local_predict)
                second_neighbour_prediction_list.append(second_neighbour_predict)
                local_estimator_list.append(local_estimator)

            return local_prediction_list, second_neighbour_prediction_list, local_estimator_list

    elif isinstance(weight_matrix, csr_array):
        def batch_wrapper(local_indices):
            local_prediction_list = []
            second_neighbour_prediction_list = []
            local_estimator_list = []
            for i in local_indices:
                estimator = estimator_list[i]
                neighbour_mask = neighbour_matrix.indices[
                                 neighbour_matrix.indptr[i]:neighbour_matrix.indptr[i + 1]
                                 ]
                second_neighbour_mask = second_neighbour_matrix.indices[
                                        second_neighbour_matrix.indptr[i]:second_neighbour_matrix.indptr[i + 1]
                                ]
                row_weight = weight_matrix.data[
                             weight_matrix.indptr[i]:weight_matrix.indptr[i + 1]
                             ]
                x = X_predict[i]
                local_predict, second_neighbour_predict, local_estimator = _fit_local_estimator(
                    estimator, X[neighbour_mask], y[neighbour_mask], local_x=x,
                    sample_weight=row_weight, X_second_neighbour=X[second_neighbour_mask],
                    return_estimator=cache_estimator
                )
                local_prediction_list.append(local_predict)
                second_neighbour_prediction_list.append(second_neighbour_predict)
                local_estimator_list.append(local_estimator)

            return local_prediction_list, second_neighbour_prediction_list, local_estimator_list

    # Split the local indices.
    local_indices_batch_list = np.array_split(local_indices, n_patches)
    parallel_batch_result = Parallel(n_patches)(
        delayed(batch_wrapper)(local_indices_batch) for local_indices_batch in local_indices_batch_list
    )

    local_predict = []
    second_neighbour_predict = []
    local_estimator_list = []
    for local_prediction_batch, second_neighbour_prediction_batch, local_estimator_batch in parallel_batch_result:
        local_predict.extend(local_prediction_batch)
        second_neighbour_predict.extend(second_neighbour_prediction_batch)
        local_estimator_list.extend(local_estimator_batch)

    if isinstance(weight_matrix, np.ndarray):
        X_meta_T = np.zeros((N, N))
        for i in range(N):
            X_meta_T[i, second_neighbour_matrix[i]] = second_neighbour_predict[i]

        X_meta = X_meta_T.T.copy()

    elif isinstance(weight_matrix, csr_array):
        X_meta_T = csr_array(
            (
                np.hstack(second_neighbour_predict),
                second_neighbour_matrix.indices,
                second_neighbour_matrix.indptr,
            )
        )
        X_meta = X_meta_T.getH().tocsr()


    t_end = time()
    logger.debug(f"Parallel fit time: {t_end - t_start}")

    return local_predict, X_meta, X_meta_T, local_estimator_list

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
        n_jobs=None,
        n_patches=None,
        alpha=10.0,
        neighbour_leave_out_rate=None,
        neighbour_leave_out_shrink_rate=None,
        meta_fitting_shrink_rate=None,
        estimator_sample_rate=None,
        use_numba=False,
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
            n_patches=n_patches,
            *args,
            **kwargs
        )
        self.alpha = alpha
        self.neighbour_leave_out_rate = neighbour_leave_out_rate
        self.neighbour_leave_out_shrink_rate = neighbour_leave_out_shrink_rate
        self.meta_fitting_shrink_rate = meta_fitting_shrink_rate
        self.estimator_sample_rate = estimator_sample_rate
        self.use_numba = use_numba

        self.base_estimator_list = None
        self.meta_estimator_list = None

        self.stacking_predict_ = None
        self.stacking_scores_ = None

        self.llocv_stacking_ = None


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
        self.log_stacking_before_fitting()

        X, y = check_X_y(X, y)
        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]
        self.N = X.shape[0]

        if coordinate_vector_list is None and weight_matrix is None:
            raise Exception('At least one of coordinate_vector_list or weight_matrix should be provided')

        # Cache data for local predict
        if self.cache_data:
            self.X = X
            self.y = y
            self.coordinate_vector_list = coordinate_vector_list
            self.coordinate_vector_dimension_ = len(coordinate_vector_list)

        cache_estimator = self.cache_estimator
        self.cache_estimator = True
        self.N = X.shape[0]

        if weight_matrix is None:
            weight_matrix = weight_matrix_from_points(
                coordinate_vector_list,
                coordinate_vector_list,
                self.distance_measure,
                self.kernel_type,
                self.distance_ratio,
                self.bandwidth,
                self.neighbour_count,
                self.distance_args,
            )

        t_neighbour_process_start = time()

        # Do the leave out neighbour sampling.
        neighbour_leave_out = None
        if self.neighbour_leave_out_rate is not None:
            neighbour_leave_out = sample_neighbour(
                weight_matrix, self.neighbour_leave_out_rate, self.neighbour_leave_out_shrink_rate
            )

            if isinstance(neighbour_leave_out, csr_array):
                neighbour_leave_out_ = neighbour_leave_out

            # From (i,j) is that i-th observation will not be used to fit the j-th base estimator
            # so that the j-th base estimator will be used for meta-estimator.
            # To (j,i) is that j-th observation will not consider i-th observation as neighbour while fitting base estimator.
            if isinstance(neighbour_leave_out, np.ndarray):
                neighbour_leave_out = neighbour_leave_out.T
            else:
                # Structure not change for sparse matrix. BUG HERE.
                neighbour_leave_out = csr_array(neighbour_leave_out.T)

        # Do not change the original weight matrix to remain the original neighbour relationship.
        # Consider the phenomenon that weight_matrix_local[neighbour_leave_out.nonzero()] is not zero?
        # Because the neighbour relationship is not symmetric.
        weight_matrix_local = weight_matrix.copy()
        weight_matrix_local[neighbour_leave_out.nonzero()] = 0
        if isinstance(weight_matrix_local, csr_array):
            # To set the value for sparse matrix, convert it first to lil_array, then convert back to csr_array.
            # This can make sure the inner structure of csr_array is correct to be able to manipulate directly .
            # Or just use eliminate_zeros() to remove the zero elements.
            weight_matrix_local.eliminate_zeros()

        if self.leave_local_out:
            if isinstance(weight_matrix_local, np.ndarray):
                np.fill_diagonal(weight_matrix_local, 0)
            else:
                # TODO: High cost for sparse matrix
                weight_matrix_local.setdiag(0)
                weight_matrix_local.eliminate_zeros()

        if self.sample_local_rate is not None:
            self.local_indices_ = np.sort(np.random.choice(self.N, int(self.sample_local_rate * self.N), replace=False))
        else:
            self.local_indices_ = range(self.N)
        self.y_sample_ = y[self.local_indices_]

        t_neighbour_process_end = time()

        if isinstance(neighbour_leave_out, np.ndarray):
            avg_neighbour_count = np.count_nonzero(weight_matrix_local) / self.N
            avg_leave_out_count = np.count_nonzero(neighbour_leave_out) / self.N
        elif isinstance(neighbour_leave_out, csr_array):
            avg_neighbour_count = weight_matrix_local.count_nonzero() / self.N
            avg_leave_out_count = neighbour_leave_out.count_nonzero() / self.N

        logger.debug(
            f"End of sampling leave out neighbour and setting weight matrix for base learner: {t_neighbour_process_end - t_neighbour_process_start}\n"
            f"Average neighbour count for fitting base learner: {avg_neighbour_count}\n"
            f"Average leave out count for fitting meta learner (n): {avg_leave_out_count}"
        )

        # Just one line of addition here to implement meta_fitting_shrink_rate.
        # TODO: BUG CHECK: the weight matrix is shrinked in place? Yes. Other operation should be checked!
        if self.meta_fitting_shrink_rate is not None:
            neighbour_shrink(weight_matrix, self.meta_fitting_shrink_rate, True)
        # weight_matrix = neighbour_shrink(weight_matrix, self.meta_fitting_shrink_rate, True)

        if isinstance(weight_matrix, np.ndarray):
            avg_neighbour_count = np.count_nonzero(weight_matrix) / self.N
        elif isinstance(weight_matrix, csr_array):
            avg_neighbour_count = weight_matrix.count_nonzero() / self.N
        logger.debug(f"End of shrinking weight matrix for meta learner. Average neighbour count for fitting meta learner (m): {avg_neighbour_count}\n")

        neighbour_matrix = weight_matrix > 0

        # Indicator of input data for each local estimator.
        # Before the local itself is set False in neighbour_matrix. Avoid no meta prediction for local.
        t_second_order_start = time()
        second_neighbour_matrix = second_order_neighbour(
            neighbour_matrix, neighbour_leave_out
        )
        t_second_order_end = time()
        logger.debug(f"End of Generating Second order neighbour matrix: {t_second_order_end - t_second_order_start}")

        if isinstance(neighbour_matrix, np.ndarray):
            np.fill_diagonal(neighbour_matrix, False)
        elif isinstance(neighbour_matrix, csr_array):
            # BUG HERE. setdiag doesn't change the structure (indptr, indices), only data change from True to False.
            neighbour_matrix.setdiag(False)
            # TO FIX: Just use eliminate_zeros
            neighbour_matrix.eliminate_zeros()

        # Iterate the stacking estimator list to get the transformed X meta.
        # Cache all the data that will be used by neighbour estimators in one iteration by using second_neighbour_matrix.
        # First dimension is data index, second dimension is estimator index.
        # X_meta[i, j] means the prediction of estimator j on data i.
        t_predict_s = time()

        t_base_fit_start = time()
        local_predict, X_meta, X_meta_T, local_estimator_list = _fit(
            X,
            y,
            estimator_list=[clone(self.local_estimator) for _ in range(self.N)],
            weight_matrix=weight_matrix_local,
            second_neighbour_matrix=second_neighbour_matrix,
            cache_estimator=True,
            n_patches=self.n_patches,
        )
        t_base_fit_end = time()

        self.local_predict_ = local_predict
        self.local_estimator_list = local_estimator_list

        self.llocv_score_ = r2_score(self.y_sample_, self.local_predict_)
        self.local_residual_ = self.y_sample_ - self.local_predict_

        self.cache_estimator = cache_estimator
        self.base_estimator_list = self.local_estimator_list
        self.local_estimator_list = None

        t_predict_e = time()
        logger.debug(f"End of predicting X_meta: {t_predict_e - t_predict_s}")

        if not self.use_numba:
            local_stacking_predict = []
            local_stacking_estimator_list = []
            indexing_time = 0
            stacking_time = 0

            if isinstance(neighbour_leave_out, np.ndarray):
                for i in range(self.N):
                    # TODO: Use RidgeCV to find best alpha
                    final_estimator = Ridge(alpha=self.alpha, solver="lsqr")

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
                        list(compress(self.base_estimator_list, neighbour_sample)),
                    )
                    local_stacking_estimator_list.append(stacking_estimator)

                    indexing_time = indexing_time + t_indexing_end - t_indexing_start
                    stacking_time = stacking_time + t_stacking_end - t_stacking_start

                self.stacking_predict_ = local_stacking_predict
                self.llocv_stacking_ = r2_score(self.y_sample_, local_stacking_predict)
                self.local_estimator_list = local_stacking_estimator_list

            elif isinstance(neighbour_leave_out, csr_array):
                for i in range(self.N):
                    final_estimator = Ridge(alpha=self.alpha, solver='lsqr')

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
                            self.base_estimator_list[leave_out_index]
                            for leave_out_index in neighbour_leave_out_indices
                        ],
                    )
                    local_stacking_estimator_list.append(stacking_estimator)

                    indexing_time = indexing_time + t_indexing_end - t_indexing_start
                    stacking_time = stacking_time + t_stacking_end - t_stacking_start

                self.stacking_predict_ = local_stacking_predict
                self.llocv_stacking_ = r2_score(self.y_sample_, local_stacking_predict)
                self.local_estimator_list = local_stacking_estimator_list

            logger.debug(f"End of fitting meta estimator without numba. Indexing/Stacking time: {indexing_time}/{stacking_time}")

        else:
            if isinstance(weight_matrix, np.ndarray):
                raise Exception("Currently, Numba not support ndarray weight matrix.")

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
                score_fit_list = [.0] * N

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

                    # TODO: If (m, n) m < n, then the matrix is not full rank, coef will be wrong.
                    coef, intercept = ridge_cholesky(X_fit_T.T, y_fit, alpha, weight_fit)

                    y_fit_predict = np.dot(X_fit_T.T, coef) + intercept
                    # TODO: Even worse, if m = 1, error will occur, the code below will be skipped in numba mode. The root cause is total_sum_squares becomes zero.
                    score_fit = r2_numba(y_fit, y_fit_predict.flatten())
                    score_fit_list[i] = score_fit

                    X_predict = np.zeros((len(leave_out_indices),))
                    for X_predict_row_index in range(len(leave_out_indices)):
                        neighbour_available_indices = X_meta_T_indices[
                            X_meta_T_indptr[
                                leave_out_indices[X_predict_row_index]
                            ] : X_meta_T_indptr[leave_out_indices[X_predict_row_index] + 1]
                        ]

                        # Find the index of the first element equals i
                        for available_iter_i in range(len(neighbour_available_indices)):
                            if neighbour_available_indices[available_iter_i] == i:
                                break

                        X_predict[X_predict_row_index] = X_meta_T_data[
                            X_meta_T_indptr[leave_out_indices[X_predict_row_index]]
                            + available_iter_i
                        ]

                    y_predict = np.dot(X_predict, coef) + intercept

                    coef_list[i] = coef.T
                    intercept_list[i] = intercept
                    y_predict_list[i] = y_predict

                return coef_list, intercept_list, y_predict_list, score_fit_list

            t_numba_start = time()
            # Different solver makes a little difference.
            coef_list, intercept_list, y_predict_list, score_fit_list = stacking_numba(
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
            t_numba_end = time()
            logger.debug("Numba running time: %s \n", t_numba_end - t_numba_start)

            self.stacking_scores_ = score_fit_list
            self.stacking_predict_ = np.array(y_predict_list)
            self.llocv_stacking_ = r2_score(self.y_sample_, self.stacking_predict_)

            self.local_estimator_list = []
            for i in range(self.N):
                final_estimator = Ridge(alpha=self.alpha, solver="cholesky")
                final_estimator.coef_ = coef_list[i]
                final_estimator.intercept_ = intercept_list[i]

                stacking_estimator = StackingEstimator(
                    final_estimator,
                    [
                        self.base_estimator_list[leave_out_index]
                        for leave_out_index in neighbour_leave_out_.indices[
                            neighbour_leave_out_.indptr[i] : neighbour_leave_out_.indptr[
                                i + 1
                            ]
                        ]
                    ],
                )

                self.local_estimator_list.append(stacking_estimator)

        # Summarize the fitting time in a single string.
        log_str = f"Leave local out elapsed: {t_neighbour_process_end - t_neighbour_process_start} \n" \
                    f"Base estimator fitting elapsed: {t_base_fit_end - t_base_fit_start} \n" \
                    f"Second order neighbour matrix elapsed: {t_second_order_end - t_second_order_start} \n" \
                    f"Meta estimator prediction elapsed: {t_predict_e - t_predict_s} \n"
        if self.use_numba:
            log_str += f"Numba running time: {t_numba_end - t_numba_start} \n"
        else:
            log_str += f"Indexing time: {indexing_time} \n" \
                    f"Stacking time: {stacking_time} \n"
        logger.debug(log_str)
        return self

    # TODO: Implement predict_by_fit

    def log_stacking_before_fitting(self):
        """
        Log the parameters about stacking before fitting.
        First, construct the log string.
        Then, log the string.
        """
        log_str = f"\nStacking Model start fitting with parameters:\n" \
                    f"alpha: {self.alpha}\n" \
                    f"neighbour_leave_out_rate: {self.neighbour_leave_out_rate}\n" \
                    f"estimator_sample_rate: {self.estimator_sample_rate}\n" \
                    f"neighbour_leave_out_shrink_rate: {self.neighbour_leave_out_shrink_rate}\n" \
                    f"meta_fitting_shrink_rate: {self.meta_fitting_shrink_rate}\n" \
                    f"use_numba: {self.use_numba}\n"

        logger.debug(log_str)
        return self


class StackingEstimator(BaseEstimator):
    def __init__(self, meta_estimator, base_estimators):
        self.meta_estimator = meta_estimator
        self.base_estimators = base_estimators

    def predict(self, X):
        X_meta = [meta_estimator.predict(X) for meta_estimator in self.base_estimators]
        X_meta = np.hstack(X_meta)
        return self.meta_estimator.predict(X_meta)
