from time import time

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone, RegressorMixin
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.inspection._partial_dependence import _grid_from_X
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_X_y
from slab_utils.quick_logger import logger
from scipy.sparse import csr_array


from georegression.weight_matrix import weight_matrix_from_points
from georegression.local_ale import weighted_ale
from georegression.ale_utils import adaptive_grid


def fit_local_estimator(
        local_estimator, X, y,
        sample_weight=None, local_x=None,
        return_estimator=False):
    """
    Wrapper for parallel fitting.
    """

    # TODO: Add partial calculation for non-cache solution.

    local_estimator.fit(X, y, sample_weight=sample_weight)

    local_predict = None
    if local_x is not None:
        local_predict = local_estimator.predict(local_x.reshape(1, -1))

    if return_estimator:
        return local_predict, local_estimator
    else:
        return local_predict, None


def _fit(X, y, estimator_list, weight_matrix,
         local_indices=None, cache_estimator=False,
         X_predict=None,
         n_jobs=None, n_patches=None):
    """
    Fit the model using provided estimators and weight matrix.

    Args:
        X:
        y:
        estimator_list:
        weight_matrix:
        local_indices:
        cache_estimator:

    Returns:

    """

    if n_jobs is not None and n_patches is not None:
        raise ValueError("Cannot specify both `n_jobs` and `n_patches`")
    if n_jobs is None and n_patches is None:
        import multiprocessing
        n_patches = multiprocessing.cpu_count()

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
        if n_jobs is not None:
            parallel_result = Parallel(n_jobs)(
                delayed(fit_local_estimator)(
                    estimator, X[neighbour_mask], y[neighbour_mask], local_x=x,
                    sample_weight=row_weight[neighbour_mask],
                    return_estimator=cache_estimator
                )
                for index, estimator, neighbour_mask, row_weight, x in
                zip(local_indices, estimator_list, neighbour_matrix, weight_matrix, X_predict)
                if index in local_indices
            )
            local_predict, local_estimator_list = list(zip(*parallel_result))
        else:
            def batch_wrapper(local_indices):
                local_prediction_list = []
                local_estimator_list = []
                for i in local_indices:
                    estimator = estimator_list[i]
                    neighbour_mask = neighbour_matrix[i]
                    row_weight = weight_matrix[i]
                    x = X_predict[i]
                    local_predict, local_estimator = fit_local_estimator(
                        estimator, X[neighbour_mask], y[neighbour_mask], local_x=x,
                        sample_weight=row_weight[neighbour_mask],
                        return_estimator=cache_estimator
                    )
                    local_prediction_list.append(local_predict)
                    local_estimator_list.append(local_estimator)

                return local_prediction_list, local_estimator_list

            local_indices_batch_list = np.array_split(local_indices, n_patches)
            parallel_batch_result = Parallel(n_patches)(
                delayed(batch_wrapper)(local_indices_batch) for local_indices_batch in local_indices_batch_list
            )

            local_predict = []
            local_estimator_list = []
            for local_prediction_batch, local_estimator_batch in parallel_batch_result:
                local_predict.extend(local_prediction_batch)
                local_estimator_list.extend(local_estimator_batch)
    else:
        # Make the task a list instead of a generator will speed up a little.
        # Use a temporal variable to save the index should speed up a lot.
        # Use native way to index also should speed up a lot.

        if n_jobs is not None:
            def task_wrapper():
                for i in local_indices:
                    estimator = estimator_list[i]
                    neighbour_mask = neighbour_matrix.indices[
                        neighbour_matrix.indptr[i]:neighbour_matrix.indptr[i + 1]
                    ]
                    row_weight = weight_matrix.data[
                        weight_matrix.indptr[i]:weight_matrix.indptr[i + 1]
                    ]
                    x = X_predict[i]
                    task = delayed(fit_local_estimator)(
                        estimator, X[neighbour_mask], y[neighbour_mask], local_x=x,
                        sample_weight=row_weight,
                        return_estimator=cache_estimator
                    )
                    yield task
            parallel_result = Parallel(n_jobs)(task_wrapper())
            local_predict, local_estimator_list = list(zip(*parallel_result))
        else:
            def batch_wrapper(local_indices):
                local_prediction_list = []
                local_estimator_list = []
                for i in local_indices:
                    estimator = estimator_list[i]
                    neighbour_mask = neighbour_matrix.indices[
                                     neighbour_matrix.indptr[i]:neighbour_matrix.indptr[i + 1]
                                     ]
                    row_weight = weight_matrix.data[
                                 weight_matrix.indptr[i]:weight_matrix.indptr[i + 1]
                                 ]
                    x = X_predict[i]
                    local_predict, local_estimator = fit_local_estimator(
                        estimator, X[neighbour_mask], y[neighbour_mask], local_x=x,
                        sample_weight=row_weight,
                        return_estimator=cache_estimator
                    )
                    local_prediction_list.append(local_predict)
                    local_estimator_list.append(local_estimator)

                return local_prediction_list, local_estimator_list

            # Split the local indices.
            local_indices_batch_list = np.array_split(local_indices, n_patches)
            parallel_batch_result = Parallel(n_patches)(
                delayed(batch_wrapper)(local_indices_batch) for local_indices_batch in local_indices_batch_list
            )

            local_predict = []
            local_estimator_list = []
            for local_prediction_batch, local_estimator_batch in parallel_batch_result:
                local_predict.extend(local_prediction_batch)
                local_estimator_list.extend(local_estimator_batch)

        # No parallel was observed. It's found later that it is probably caused by the indexing and transferring bandwidth,
        # because it only happens when the data is large.
        # TODO: Manually split the task into batches and use parallel to speed up.

    t_end = time()
    logger.debug(f"Parallel fit time: {t_end - t_start}")

    return local_predict, local_estimator_list


def local_partial_dependence(local_estimator, X_local, weight, n_features_in_):
    """
    Wrapper for parallel partial dependence calculation
    """

    # Partial result of each feature.
    # [(x for feature1, y for feature1), (x for feature2, y for feature2), (...), ...]
    feature_list = []

    for feature_index in range(n_features_in_):
        pdp = partial_dependence(
            local_estimator,
            X_local,
            [feature_index],
            kind='both'
        )

        values = pdp['values'][0]
        individual = pdp['individual'][0]

        # Must get individual partial dependence to weight the result
        # Weight: Performance Weight. The point with more weight performance better in the model.
        # So average the partial performance according to the weight.
        weight_average = np.average(individual, axis=0, weights=weight)

        feature_list.append((values, weight_average))

    return feature_list


def local_importance(local_estimator, X_local, y_local, weight, n_repeats=5):
    """
    Wrapper for parallel local importance calculation
    """
    importance_result = permutation_importance(
        local_estimator,
        X_local,
        y_local,
        sample_weight=weight,
        n_repeats=n_repeats,
    )

    print('One local finished')
        
    return importance_result.importances_mean


class WeightModel(BaseEstimator, RegressorMixin):
    """
    Inherit from sklearn BaseEstimator to support sklearn workflow, e.g. GridSearchCV.
    """

    def __init__(self,
                 local_estimator,
                 # Weight matrix param
                 distance_measure=None,
                 kernel_type=None,
                 distance_ratio=None,
                 bandwidth=None,
                 neighbour_count=None,
                 distance_args=None,
                 # Model param
                 leave_local_out=True,
                 sample_local_rate=None,
                 cache_data=False,
                 cache_estimator=False,
                 n_jobs=None, n_patches=None,
                 *args, **kwargs):

        # Parameters of the model
        self.local_estimator = local_estimator
        self.distance_measure = distance_measure
        self.kernel_type = kernel_type
        self.distance_ratio = distance_ratio
        self.bandwidth = bandwidth
        self.neighbour_count = neighbour_count
        self.distance_args = distance_args
        self.leave_local_out = leave_local_out
        self.sample_local_rate = sample_local_rate
        self.cache_data = cache_data
        self.cache_estimator = cache_estimator
        self.n_jobs = n_jobs
        if n_jobs is None and n_patches is None:
            import multiprocessing
            n_patches = multiprocessing.cpu_count()
        self.n_patches = n_patches

        # Attributes of the model
        self.is_fitted_ = None
        self.n_features_in_ = None

        self.X = None
        self.y = None
        self.N = None
        self.coordinate_vector_list = None
        self.coordinate_vector_dimension_ = None

        self.weight_matrix_ = None
        self.neighbour_matrix_ = None
        self.local_indices_ = None
        self.y_sample_ = None

        self.local_estimator_list = None
        self.local_predict_ = None
        # Leave local out cross validation score
        self.llocv_score_ = None
        self.local_residual_ = None

        # Permutation Importance
        self.permutation_score_decrease_ = None
        self.interaction_matrix_ = None
        # Partial Dependence
        self.local_partial_ = None
        self.feature_partial_ = None
        # ICE
        self.local_ice_ = None
        self.feature_ice_ = None

        self.args = args
        self.kwargs = kwargs


    def fit(self, X, y, coordinate_vector_list=None, weight_matrix=None):
        """
        Fix the model

        Args:
            X:
            y:
            coordinate_vector_list:
            weight_matrix:

        Returns:

        """
        self.log_before_fitting(X, y, coordinate_vector_list, weight_matrix)

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

        if weight_matrix is None:
            weight_matrix = weight_matrix_from_points(coordinate_vector_list, coordinate_vector_list, self.distance_measure, self.kernel_type, self.distance_ratio, self.bandwidth,
                                                      self.neighbour_count, self.distance_args)

        # TODO: Tweak for inspection.
        self.weight_matrix_ = weight_matrix
        # Set the diagonal value of the weight matrix to exclude the local location to get CV score
        if self.leave_local_out:
            if isinstance(self.weight_matrix_, np.ndarray):
                np.fill_diagonal(self.weight_matrix_, 0)
            else:
                # TODO: High cost for sparse matrix
                self.weight_matrix_.setdiag(0)
                self.weight_matrix_.eliminate_zeros()

        self.neighbour_matrix_ = self.weight_matrix_ > 0

        # TODO: Repeatable randomness of sklearn style.
        # Sample Procedure. Fit local models for only a part of points to reduce computation.
        if self.sample_local_rate is not None:
            self.local_indices_ = np.sort(np.random.choice(self.N, int(self.sample_local_rate * self.N), replace=False))
        else:
            self.local_indices_ = range(self.N)
        self.y_sample_ = y[self.local_indices_]

        # Clone local estimators
        estimator_list = [clone(self.local_estimator) for _ in range(self.N)]

        # Fit
        # Embed the local predict in the parallel build process to speed up.
        # Reduce the expense of resource allocation between processes.
        self.local_predict_, self.local_estimator_list = _fit(
            X, y, estimator_list, self.weight_matrix_, self.local_indices_,
            cache_estimator=self.cache_estimator, n_jobs=self.n_jobs, n_patches=self.n_patches
        )
        self.local_predict_ = np.array(self.local_predict_).squeeze()

        # Calculate the CV score and other metrics according the local result
        self.llocv_score_ = r2_score(self.y_sample_, self.local_predict_)
        self.local_residual_ = self.y_sample_ - self.local_predict_

        return self

    def predict_by_weight(self, X, coordinate_vector_list=None, weight_matrix=None,
                          search_optimal=False, y=None, *args, **kwargs):
        """
        Predict using local estimators.
        Weight the local predictions using the weight matrix which uses input X as source and fitted X as target.

        Args:
            X ():
            coordinate_vector_list ():
            weight_matrix ():
            search_optimal ():
            y ():
            *args ():
            **kwargs ():

        Returns:

        """

        if not self.cache_data or not self.cache_estimator:
            raise Exception('Prediction by weight need cache_data and cache_estimator set True')

        if coordinate_vector_list is None and weight_matrix:
            raise Exception('At least one of coordinate_vector_list or weight_matrix should be provided')

        # Parallel will slow down the process for the reason of data allocation between processes.
        local_predict = np.vstack([local_estimator.predict(X) for local_estimator in self.local_estimator_list])

        # Search the best parameters for prediction.
        if search_optimal:
            if y is None:
                raise Exception('y needs to be provided to search the best parameter for prediction')
            # set the best weight matrix for prediction.
            pass

        if weight_matrix is None:
            weight_matrix = weight_matrix_from_points(coordinate_vector_list, self.coordinate_vector_list, self.distance_measure, self.kernel_type, self.distance_ratio, self.bandwidth,
                                                      self.neighbour_count, self.distance_args)

        return np.sum(weight_matrix * local_predict.T, axis=1)

    def predict_by_fit(self, X, coordinate_vector_list=None, weight_matrix=None, *args, **kwargs):
        """
        Fit new local model for prediction data using the training data to make prediction.
        Calculate weight matrix by using predicting X as source and training X as target.

        Args:
            X ():
            coordinate_vector_list ():
            weight_matrix ():
            *args ():
            **kwargs ():

        Returns:

        """

        if not self.cache_data:
            raise Exception('Prediction by fit need cache_data set True')

        if coordinate_vector_list is None and weight_matrix:
            raise Exception('At least one of coordinate_vector_list or weight_matrix should be provided')

        if weight_matrix is None:
            # weight_matrix = weight_matrix_from_points(coordinate_vector_list, self.coordinate_vector_list,
            #                                           self.distance_measure, self.kernel_type, self.distance_ratio,
            #                                           self.bandwidth, self.neighbour_count, self.distance_args)
            weight_matrix = weight_matrix_from_points(self.coordinate_vector_list, coordinate_vector_list,
                                                      self.distance_measure, self.kernel_type, self.distance_ratio,
                                                      self.bandwidth, self.neighbour_count, self.distance_args)
            weight_matrix = weight_matrix.T.copy()

        N = X.shape[0]

        # Parallel build RandomForest
        estimator_list = [clone(self.local_estimator) for _ in range(N)]

        # Train local model using training set
        # but with weight matrix constructed from distance between prediction points and training point for prediction.
        local_predict, _ = _fit(self.X, self.y, estimator_list, weight_matrix, X_predict=X)

        return local_predict

    def importance_score_local(self, n_repeats=5, n_jobs=-1):
        """
        Calculate the importance score using permutation test (MDA, Mean decrease accuracy) for each local estimators.

        Args:
            n_repeats ():

        Returns: Shape(N, Feature). Each row represents a local estimator, each column represents a feature.

        """
        # TODO: Consider use out-of-bag sample for permutation importance to improve the ability of generalization.
        # TODO: Weight of the OOB error/score?

        if not self.cache_data or not self.cache_estimator:
            raise Exception('Importance score of local needs cache_data and cache_estimator set True')

        job_list = []
        for local_index in range(self.N):
            local_estimator = self.local_estimator_list[local_index]
            weight = self.weight_matrix_[local_index]
            neighbour_mask = self.neighbour_matrix_[local_index]

            job_list.append(
                delayed(local_importance)(
                    local_estimator, self.X[neighbour_mask], self.y[neighbour_mask], weight[neighbour_mask], n_repeats
                )
            )

        importance_matrix = Parallel(n_jobs=n_jobs)(job_list)
        importance_matrix = np.array(importance_matrix)

        return importance_matrix

    def importance_score_global(self, n_repeats=5):
        """
        Permute the X and predict the local value for each local estimator using the permuted data.

        Args:
            n_repeats ():

        Returns:
            Return average of each Feature.
            permutation_score_decrease_: Shape(Feature, n_repeats).

        """
        if not self.cache_data or not self.cache_estimator:
            raise Exception('Importance score of global needs cache_data and cache_estimator set True')

        # Feed bulk/batch instead of atomic data to local_estimator to speed up 100x.

        # Prediction after permuting. Shape(N, Feature, n_repeats)
        permute_predict = np.empty((self.N, self.n_features_in_, n_repeats))

        # Index of permutation along the N axis. Shape(N, Feature, n_repeats)
        permute_index = np.tile(np.arange(self.N).reshape((-1, 1, 1)), (1, self.n_features_in_, n_repeats))
        permute_index = np.apply_along_axis(np.random.permutation, 0, permute_index)

        # Predict for each local estimator
        for i in range(self.N):
            local_estimator = self.local_estimator_list[i]

            # Input of the estimator. Shape(Feature, Feature, n_repeats)
            x = self.X[i].reshape(-1, 1, 1)
            x = np.tile(x, (1, self.n_features_in_, n_repeats))

            # Permute the x using permute_index. Iterate by feature.
            for feature_index in range(self.n_features_in_):
                x[feature_index, feature_index, :] = self.X[permute_index[i, feature_index, :], feature_index]

            # Flatten and transpose the x for estimation input. Shape(Feature * n_repeats, Feature)
            x = np.transpose(x.reshape(self.n_features_in_, -1))

            # Predict and reshape back. Shape(Feature, n_repeats) corresponding to "Feature * n_repeats".
            permute_predict[i, :, :] = local_estimator.predict(x).reshape(self.n_features_in_, n_repeats)

        def inner_score(y_hat):
            return r2_score(self.y, y_hat)

        # The lower score means higher importance.
        score_decrease = self.llocv_score_ - np.apply_along_axis(inner_score, 0, permute_predict)

        # More trail added.
        if self.permutation_score_decrease_ is not None:
            self.permutation_score_decrease_ = np.concatenate(
                [self.permutation_score_decrease_, score_decrease], axis=1
            )
        else:
            self.permutation_score_decrease_ = score_decrease

        importance_score = np.average(self.permutation_score_decrease_, axis=1)
        return importance_score

    def interaction_score_global(self, n_repeats=5):
        """
        Interaction importance of feature-pair {i,j} = Importance of {i,j} - Importance of i - Importance of j.
        References to SHAP Interation.

        Args:
            n_repeats ():

        Returns: Interaction matrix averaged over n_repeats. Shape(Feature, Feature).

        """
        if not self.cache_data or not self.cache_estimator:
            raise Exception('Importance score of global needs cache_data and cache_estimator set True')

        # Single feature score decrease
        if self.permutation_score_decrease_ is None:
            self.importance_score_global()
        single_score_decrease = np.average(self.permutation_score_decrease_, axis=1)

        # Shape(Feature, Feature, n_repeats)
        interaction_matrix = np.empty((self.n_features_in_, self.n_features_in_, n_repeats))
        for first_feature_index in range(self.n_features_in_):
            for second_feature_index in range(first_feature_index + 1, self.n_features_in_):

                # Shape(n_repeats, )
                feature_score_decrease_list = []

                # TODO: Parallel or bulk/batch data to speed up.
                for repeat in range(n_repeats):
                    shuffling_idx = np.arange(self.N)
                    X_permuted = self.X.copy()

                    np.random.shuffle(shuffling_idx)
                    X_permuted[:, first_feature_index] = X_permuted[shuffling_idx, first_feature_index]
                    np.random.shuffle(shuffling_idx)
                    X_permuted[:, second_feature_index] = X_permuted[shuffling_idx, second_feature_index]

                    # Predict local using the permuted data
                    local_predict = [
                        local_estimator.predict(x.reshape(1, -1))
                        for local_estimator, x in
                        zip(self.local_estimator_list, X_permuted)
                    ]
                    local_predict = np.array(local_predict).squeeze()

                    score = r2_score(self.y, local_predict)
                    score_decrease = self.llocv_score_ - score
                    feature_score_decrease_list.append(score_decrease)

                feature_score_decrease = np.array(feature_score_decrease_list)
                feature_score_interaction = feature_score_decrease \
                                            - single_score_decrease[first_feature_index] \
                                            - single_score_decrease[second_feature_index]

                interaction_matrix[first_feature_index, second_feature_index] = feature_score_interaction

        interaction_matrix = interaction_matrix + interaction_matrix.T
        if self.interaction_matrix_ is not None:
            self.interaction_matrix_ = np.concatenate([self.interaction_matrix_, interaction_matrix], axis=2)
        else:
            self.interaction_matrix_ = interaction_matrix

        interaction_score = np.average(self.interaction_matrix_, axis=2)
        return interaction_score

    def partial_dependence(self):
        # TODO: More detailed reason to justify the weighted partial dependence.
        # Care more on the local range.
        # Unweighted points will dominate the tendency which may not be the interested one.

        # Partial feature list for each local estimator.
        # [feature_list for estimator1, ...2, ...]
        local_partial_list = []

        # TODO: Batch optimize for partial_dependence. Native weighted partial dependence.
        job_list = []
        for local_index in range(self.N):
            local_estimator = self.local_estimator_list[local_index]
            weight = self.weight_matrix_[local_index][self.neighbour_matrix_[local_index]]
            X_local = self.X[self.neighbour_matrix_[local_index]]

            job_list.append(
                delayed(local_partial_dependence)(local_estimator, X_local, weight, self.n_features_in_)
            )

        local_partial_list = Parallel(n_jobs=-1)(job_list)
        self.local_partial_ = np.array(local_partial_list, dtype=object)

        # Convert local based result to feature based result.
        '''
        [
            Feature1: [(x for estimator1, y for estimator1), (x for estimator2, y for estimator2), (...), ...],
            Feature2: [...],
            ...
        ]
        '''
        # self.feature_partial_list_ = list(zip(*self.local_partial_list_))
        self.feature_partial_ = self.local_partial_.transpose((1, 0, 2))

        return self.local_partial_

    def local_ICE(self):
        local_ice_list = []
        for local_index in range(self.N):
            X_local = self.X[self.neighbour_matrix_[local_index]]
            local_estimator = self.local_estimator_list[local_index]
            percentiles = (0.05, 0.95)
            grid_resolution = 100

            feature_list = []
            for feature_index in range(self.n_features_in_):
                grid, values = _grid_from_X(X_local[:, [feature_index]], percentiles, grid_resolution)
                values = values[0]
                X_individual = np.tile(self.X[local_index], (len(values), 1))
                X_individual[:, feature_index] = values
                y = local_estimator.predict(X_individual)

                feature_list.append((values, y))

            local_ice_list.append(feature_list)

        self.local_ice_ = np.array(local_ice_list, dtype=object)
        self.feature_ice_ = self.local_ice_.transpose((1, 0, 2))

        return self.local_ice_

    def local_ALE(self, feature=0):
        ale_list = []

        for local_index in range(len(self.local_estimator_list)):
            estimator = self.local_estimator_list[local_index]
            neighbour_mask = self.neighbour_matrix_[local_index]
            neighbour_weight = self.weight_matrix_[local_index][neighbour_mask]
            X_local = self.X[neighbour_mask]
            ale_result = weighted_ale(X_local, feature, estimator.predict, neighbour_weight)
            ale_list.append(ale_result)

        return ale_list


    def global_ALE(self, feature=0):
        fvals, _ = adaptive_grid(self.X[:, feature])

        # find which interval each observation falls into
        indices = np.searchsorted(fvals, self.X[:, feature], side="left")
        indices[indices == 0] = 1  # put the smallest data point in the first interval
        interval_n = np.bincount(indices)  # number of points in each interval

        # predictions for the upper and lower ranges of intervals
        z_low = self.X.copy()
        z_high = self.X.copy()
        z_low[:, feature] = fvals[indices - 1]
        z_high[:, feature] = fvals[indices]

        p_low_list = []
        p_high_list = []
        for i in range(len(self.local_estimator_list)):
            local_estimator = self.local_estimator_list[i]
            p_local = local_estimator.predict(np.vstack((z_low[[i], :], z_high[[i], :])))
            p_low_local = p_local[0]
            p_high_local = p_local[1]
            p_low_list.append(p_low_local)
            p_high_list.append(p_high_local)

        p_low = np.array(p_low_list)
        p_high = np.array(p_high_list)

        # finite differences
        p_deltas = p_high - p_low

        # make a dataframe for averaging over intervals
        concat = np.column_stack((p_deltas, indices))
        df = pd.DataFrame(concat)

        # weighted average for each interval
        avg_p_deltas = df.groupby(1).apply(lambda x: np.average(x[0])).values

        # accumulate over intervals
        accum_p_deltas = np.cumsum(avg_p_deltas, axis=0)

        # pre-pend 0 for the left-most point
        zeros = np.zeros((1, 1))
        accum_p_deltas = np.insert(accum_p_deltas, 0, zeros, axis=0)

        # mean effect, R's `ALEPlot` and `iml` version (approximation per interval)
        # Eq.16 from original paper "Visualizing the effects of predictor variables in black box supervised learning models"
        ale0 = (
                0.5 * (accum_p_deltas[:-1] + accum_p_deltas[1:]) * interval_n[1:]
        ).sum(axis=0)
        ale0 = ale0 / interval_n.sum()

        # center
        ale = accum_p_deltas - ale0

        return fvals, ale


    def log_before_fitting(self, X, y, coordinate_vector_list=None, weight_matrix=None):
        """
        Log the parameters before fitting.
        First, construct the log string.
        Then, log the string.
        """
        log_str = f"\nWeight Model start fitting with data:\n" \
                    f"X.shape: {X.shape}\n"

        log_str += f"\nWeight Model start fitting with parameters:\n" \
                    f"local_estimator: {self.local_estimator}\n" \
                    f"leave_local_out: {self.leave_local_out}\n" \
                    f"sample_local_rate: {self.sample_local_rate}\n" \
                    f"cache_data: {self.cache_data}\n" \
                    f"cache_estimator: {self.cache_estimator}\n" \
                    f"n_jobs: {self.n_jobs}\n" \
                    f"n_patches: {self.n_patches}\n" \
                    f"args: {self.args}\n" \
                    f"kwargs: {self.kwargs}\n"

        # Should not appear here? In weight matrix instead?
        # if coordinate_vector_list is not None:
        #     log_str += f"Coordinate Dimension: {len(coordinate_vector_list)}\n"
        #     for i, coordinate_vector in enumerate(coordinate_vector_list):
        #         log_str +=  f"coordinate_vector[{i}].shape: {coordinate_vector.shape}\n"
        #
        # log_str += f"\nWeight Matrix Info:\n"
        # if weight_matrix is not None:
        #     log_str += f"weight_matrix.shape: {weight_matrix.shape}\n"
        # else:
        #     log_str += f"distance_measure: {self.distance_measure}\n" \
        #             f"kernel_type: {self.kernel_type}\n" \
        #             f"distance_ratio: {self.distance_ratio}\n" \
        #             f"bandwidth: {self.bandwidth}\n" \
        #             f"neighbour_count: {self.neighbour_count}\n" \
        #             f"distance_args: {self.distance_args}\n"

        logger.debug(log_str)
        return self







