from time import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone, RegressorMixin
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.inspection._partial_dependence import _grid_from_X
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_X_y
from slab_utils.quick_logger import logger

from georegression.weight_matrix import calculate_compound_weight_matrix


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
         n_jobs=-1):
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

    t_start = time()
    logger.debug(f'Fit begin {t_start}')

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

    t_end = time()
    logger.debug(f'Fit end {t_end}. {t_end - t_start} elapsed')

    return local_predict, local_estimator_list


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
                 midpoint=None,
                 p=None,
                 # Model param
                 leave_local_out=True,
                 sample_local_rate=None,
                 cache_data=False,
                 cache_estimator=False,
                 n_jobs=-1,
                 *args, **kwargs):

        # Parameters of the model
        self.local_estimator = local_estimator
        self.distance_measure = distance_measure
        self.kernel_type = kernel_type
        self.distance_ratio = distance_ratio
        self.bandwidth = bandwidth
        self.neighbour_count = neighbour_count
        self.midpoint = midpoint
        self.p = p
        self.leave_local_out = leave_local_out
        self.sample_local_rate = sample_local_rate
        self.cache_data = cache_data
        self.cache_estimator = cache_estimator
        self.n_jobs = n_jobs

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

        if weight_matrix is None:
            weight_matrix = calculate_compound_weight_matrix(coordinate_vector_list,
                                                             coordinate_vector_list,
                                                             self.distance_measure,
                                                             self.kernel_type,
                                                             self.distance_ratio,
                                                             self.bandwidth,
                                                             self.neighbour_count,
                                                             self.midpoint,
                                                             self.p)
        self.weight_matrix_ = weight_matrix
        # Set the diagonal value of the weight matrix to exclude the local location to get CV score
        if self.leave_local_out:
            np.fill_diagonal(self.weight_matrix_, 0)

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
            cache_estimator=self.cache_estimator
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
            weight_matrix = calculate_compound_weight_matrix(coordinate_vector_list, self.coordinate_vector_list,
                                                             self.distance_measure,
                                                             self.kernel_type,
                                                             self.distance_ratio,
                                                             self.bandwidth,
                                                             self.neighbour_count,
                                                             self.midpoint,
                                                             self.p)

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
            weight_matrix = calculate_compound_weight_matrix(coordinate_vector_list, self.coordinate_vector_list,
                                                             self.distance_measure,
                                                             self.kernel_type,
                                                             self.distance_ratio,
                                                             self.bandwidth,
                                                             self.neighbour_count,
                                                             self.midpoint,
                                                             self.p)

        N = X.shape[0]

        # Parallel build RandomForest
        estimator_list = [clone(self.local_estimator) for _ in range(N)]

        # Train local model using training set
        # but with weight matrix constructed from distance between prediction points and training point for prediction.
        local_predict, _ = _fit(self.X, self.y, estimator_list, weight_matrix, X_predict=X)

        return local_predict

    def importance_score_local(self, n_repeats=5):
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

        importance_matrix = []
        for local_index, local_estimator in enumerate(self.local_estimator_list):
            weight = self.weight_matrix_[local_index]
            neighbour_mask = self.neighbour_matrix_[local_index]

            # Select X to speed up calculation
            importance_result = permutation_importance(
                local_estimator,
                self.X[neighbour_mask],
                self.y[neighbour_mask],
                sample_weight=weight[neighbour_mask],
                n_repeats=n_repeats,
                n_jobs=self.n_jobs
            )
            importance_matrix.append(importance_result.importances_mean)
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

        # Shape(Feature, n_repeats)
        score_list = []
        for feature_index in range(self.n_features_in_):

            # Shape(n_repeats, )
            feature_score_list = []

            # TODO: Parallel or bulk/batch data to speed up.
            for repeat in range(n_repeats):
                # Permutation the feature
                shuffling_idx = np.arange(self.N)
                np.random.shuffle(shuffling_idx)
                X_permuted = self.X.copy()
                X_permuted[:, feature_index] = self.X[shuffling_idx, feature_index]

                # Predict local using the permuted data
                local_predict = [
                    local_estimator.predict(x.reshape(1, -1))
                    for local_estimator, x in
                    zip(self.local_estimator_list, X_permuted)
                ]
                local_predict = np.array(local_predict).squeeze()

                # Get score and append to the current feature score list
                score = r2_score(self.y, local_predict)
                feature_score_list.append(score)

            # N-repeat scores of each feature
            score_list.append(feature_score_list)

        # The lower score means higher importance.
        score_decrease = self.llocv_score_ - np.array(score_list)

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

        Returns:

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
        for local_index in range(self.N):
            local_estimator = self.local_estimator_list[local_index]
            weight = self.weight_matrix_[local_index][self.neighbour_matrix_[local_index]]
            X_local = self.X[self.neighbour_matrix_[local_index]]

            # Partial result of each feature.
            # [(x for feature1, y for feature1), (x for feature2, y for feature2), (...), ...]
            feature_list = []

            for feature_index in range(self.n_features_in_):
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

            local_partial_list.append(feature_list)
        self.local_partial_ = np.array(local_partial_list)

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

        self.local_ice_ = np.array(local_ice_list)
        self.feature_ice_ = self.local_ice_.transpose((1, 0, 2))

        return self.local_ice_
