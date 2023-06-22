import math
from time import time

import numpy as np
from numba import njit, prange
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from slab_utils.quick_logger import logger

from georegression.weight_model import WeightModel


@njit()
def second_order_neighbour(neighbour_matrix):
    """
    Calculate second-order neighbour matrix.
    Args:
        neighbour_matrix: First-order neighbour matrix

    Returns:

    """
    second_order_matrix = np.empty_like(neighbour_matrix)
    for i in prange(neighbour_matrix.shape[0]):
        second_order_matrix[i] = np.sum(neighbour_matrix[neighbour_matrix[i]], axis=0)
    return second_order_matrix


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
        p=None,
        # Model param
        leave_local_out=True,
        sample_local_rate=None,
        cache_data=False,
        cache_estimator=False,
        n_jobs=-1,
        alpha=10,
        estimator_sample_rate=None,
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
            p=p,
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
        self.estimator_sample_rate = estimator_sample_rate

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

        t_local_start = time()

        super().fit(
            X,
            y,
            coordinate_vector_list=coordinate_vector_list,
            weight_matrix=weight_matrix,
        )

        t_local_end = time()
        logger.debug("Local model fitting elapsed: %s", t_local_end - t_local_start)

        self.cache_estimator = cache_estimator
        self.meta_estimator_list = np.array(self.local_estimator_list)
        self.local_estimator_list = None

        neighbour_matrix = self.weight_matrix_ > 0
        weight_matrix = self.weight_matrix_

        # TODO: Add parallel
        # Indicator of input data for each local estimator.
        t_second_order_start = time()
        second_neighbour_matrix = second_order_neighbour(neighbour_matrix)
        t_second_order_end = time()
        logger.debug(
            "Second order neighbour matrix elapsed: %s",
            t_second_order_end - t_second_order_start,
        )

        # Iterate the stacking estimator list to get the transformed X meta. First dimension is data index, second dimension is estimator index.
        # X_meta[i, j] means the prediction of estimator j on data i.
        t_predict_s = time()
        X_meta = np.zeros((self.N, self.N))
        for i in range(self.N):
            X_meta[second_neighbour_matrix[i], i] = self.meta_estimator_list[i].predict(
                X[second_neighbour_matrix[i]]
            )
        t_predict_e = time()
        logger.debug("Meta estimator prediction elapsed: %s", t_predict_e - t_predict_s)

        t_transpose_start = time()
        X_meta_T = X_meta.transpose().copy(order="C")
        t_transpost_end = time()
        print("Transpose time: ", t_transpost_end - t_transpose_start)

        # TODO: Add parallel
        local_stacking_predict = []
        local_stacking_estimator_list = []
        indexing_time = 0
        stacking_time = 0
        for i in range(self.N):
            # TODO: Use RidgeCV to find best alpha
            final_estimator = Ridge(alpha=self.alpha, solver="lsqr")

            # TODO: Consider whether to add the meta prediction of the local meta estimator.
            t_indexing_start = time()

            # Sample from neighbour bool matrix to get sampled neighbour index.
            if self.estimator_sample_rate is not None:
                neighbour_indexes = np.nonzero(neighbour_matrix[i])
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
            else:
                neighbour_sample = neighbour_matrix[i]

            # X_fit = X_meta[:, neighbour_sample][neighbour_matrix[i]]
            # X_fit = X_meta[neighbour_matrix[i], :][:, neighbour_sample]
            # X_fit = X_meta[
            #     np.matmul(
            #         neighbour_matrix[i].reshape(-1, 1), neighbour_sample.reshape(1, -1)
            #     )
            # ].reshape(-1, neighbour_sample.sum())
            # X_fit = X_meta.T[neighbour_sample][:, neighbour_matrix[i]].T
            # X_fit = X_meta[neighbour_matrix[i]][:, neighbour_sample]
            X_fit = X_meta_T[neighbour_sample][:, neighbour_matrix[i]].T
            y_fit = y[neighbour_matrix[i]]
            t_indexing_end = time()

            t_stacking_start = time()
            final_estimator.fit(
                X_fit, y_fit, sample_weight=weight_matrix[i, neighbour_matrix[i]]
            )
            t_stacking_end = time()

            local_stacking_predict.append(
                final_estimator.predict(np.expand_dims(X_meta[i, neighbour_sample], 0))
            )
            stacking_estimator = StackingEstimator(
                final_estimator, self.meta_estimator_list[neighbour_sample]
            )
            local_stacking_estimator_list.append(stacking_estimator)

            indexing_time = indexing_time + t_indexing_end - t_indexing_start
            stacking_time = stacking_time + t_stacking_end - t_stacking_start

        logger.debug("Indexing time: %s", indexing_time)
        logger.debug("Stacking time: %s", stacking_time)

        # def stacking_job(X_fit, Y_fit, sample_weight, X_local):
        #     final_estimator = Ridge(alpha=self.alpha, solver="lsqr")
        #     final_estimator.fit(X_fit, Y_fit, sample_weight=sample_weight)
        #     return final_estimator, final_estimator.predict(X_local)
        #
        # t_parallel_indexing_start = time()
        # job_list = []
        # for i in range(self.N):
        #     job_list.append(
        #         delayed(stacking_job)(
        #             X_meta[neighbour_matrix[i]][:, neighbour_matrix[i]],
        #             y[neighbour_matrix[i]],
        #             weight_matrix[i, neighbour_matrix[i]],
        #             np.expand_dims(X_meta[i, neighbour_matrix[i]], 0),
        #         )
        #     )
        # t_parallel_indexing_end = time()
        # logger.debug(
        #     "Parallel indexing elapsed: %s",
        #     t_parallel_indexing_end - t_parallel_indexing_start,
        # )
        #
        # t_parallel_start = time()
        # parallel_result = Parallel(n_jobs=self.n_jobs)(job_list)
        # for i in range(self.N):
        #     local_stacking_estimator_list.append(parallel_result[i][0])
        #     local_stacking_predict.append(parallel_result[i][1])
        # t_parallel_end = time()
        # logger.debug("Parallel stacking elapsed: %s", t_parallel_end - t_parallel_start)

        self.stacking_predict_ = local_stacking_predict
        self.llocv_stacking_ = r2_score(self.y_sample_, local_stacking_predict)
        self.local_estimator_list = local_stacking_estimator_list

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
