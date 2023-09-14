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
from joblib import Parallel, delayed

from georegression.neighbour_utils import second_order_neighbour, sample_neighbour
from georegression.numba_impl import ridge_cholesky
from georegression.weight_matrix import calculate_compound_weight_matrix
from georegression.weight_model import WeightModel
from georegression.numba_impl import r2_score as r2_numba


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
            n_patches=n_patches,
            *args,
            **kwargs
        )

        self.base_estimator_list = None
        self.stacking_predict_ = None
        self.llocv_stacking_ = None
        self.fitting_score_stacking_ = None
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

        t_neighbour_end = time()
        logger.debug("End of Neighbour leave out")

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
        self.base_estimator_list = self.local_estimator_list
        self.local_estimator_list = None

        # Iterate the stacking estimator list to get the transformed X meta.
        # Cache all the data that will be used by neighbour estimators in one iteration by using second_neighbour_matrix.
        # First dimension is data index, second dimension is estimator index.
        # X_meta[i, j] means the prediction of estimator j on data i.
        t_predict_s = time()

        # TODO: Parallelize this part.
        if isinstance(second_neighbour_matrix, np.ndarray):
            X_meta = np.zeros((self.N, self.N))
            for i in range(self.N):
                if second_neighbour_matrix[i].any():
                    X_meta[second_neighbour_matrix[i], i] = self.base_estimator_list[
                        i
                    ].predict(X[second_neighbour_matrix[i]])
        elif isinstance(second_neighbour_matrix, csr_array):
            prediction_result = list()
            for i in range(self.N):
                if (
                    second_neighbour_matrix.indptr[i]
                    != second_neighbour_matrix.indptr[i + 1]
                ):
                    prediction_result.append(
                        self.base_estimator_list[i].predict(
                            X[
                                second_neighbour_matrix.indices[
                                    second_neighbour_matrix.indptr[
                                        i
                                    ] : second_neighbour_matrix.indptr[i + 1]
                                ]
                            ]
                        )
                    )

            # TODO: Why it failed to accelerate?
            # def batch_wrapper(indices_batch, base_estimator_batch, second_neighbour_matrix, X):
            #     prediction_result_batch = list()
            #     for estimator_index ,i in enumerate(indices_batch):
            #         if (
            #             second_neighbour_matrix.indptr[i]
            #             != second_neighbour_matrix.indptr[i + 1]
            #         ):
            #             prediction_result_batch.append(
            #                 base_estimator_batch[estimator_index].predict(
            #                     X[
            #                         second_neighbour_matrix.indices[
            #                             second_neighbour_matrix.indptr[
            #                                 i
            #                             ] : second_neighbour_matrix.indptr[i + 1]
            #                         ]
            #                     ]
            #                 )
            #             )
            #     return prediction_result_batch
            #
            # indices_list = np.array_split(range(self.N), int(self.n_patches / 4))
            # parallel_batch_result = Parallel(int(self.n_patches / 4))(list(
            #     delayed(batch_wrapper)(
            #         indices_batch,
            #         self.base_estimator_list[indices_batch[0]: indices_batch[-1] + 1],
            #         second_neighbour_matrix,
            #         X
            #     )
            #     for indices_batch in indices_list
            # ))
            # prediction_result = [
            #     prediction_result
            #     for batch_result in parallel_batch_result
            #     for prediction_result in batch_result
            # ]

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

        if isinstance(X_meta, np.ndarray):
            X_meta_T = X_meta.transpose().copy(order="C")


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

        elif isinstance(neighbour_leave_out, csr_array) and not self.use_numba:
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

                    coef, intercept = ridge_cholesky(X_fit_T.T, y_fit, alpha, weight_fit)

                    y_fit_predict = np.dot(X_fit_T.T, coef) + intercept
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

            t1 = time()
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
            t2 = time()
            logger.debug("Numba running time: %s \n", t2 - t1)

            self.stacking_predict_ = np.array(y_predict_list)
            self.fitting_score_stacking_ = score_fit_list
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

                local_stacking_estimator_list.append(stacking_estimator)

        # Log the time elapsed in a single line
        logger.debug(
            "Leave local out elapsed: %s \n"
            "Second order neighbour matrix elapsed: %s \n"
            "Meta estimator prediction elapsed: %s \n"
            "Indexing time: %s \n"
            "Stacking time: %s \n",
            t_neighbour_end - t_neighbour_start,
            t_second_order_end - t_second_order_start,
            t_predict_e - t_predict_s,
            indexing_time,
            stacking_time,
        )

        return self

    # TODO: Implement predict_by_fit


class StackingEstimator(BaseEstimator):
    def __init__(self, meta_estimator, base_estimators):
        self.meta_estimator = meta_estimator
        self.base_estimators = base_estimators

    def predict(self, X):
        X_meta = [meta_estimator.predict(X) for meta_estimator in self.base_estimators]
        X_meta = np.hstack(X_meta)
        return self.meta_estimator.predict(X_meta)
