from os.path import join

import numpy as np
from sklearn.base import RegressorMixin

from georegression.visualize.scatter import scatter_3d
from georegression.weight_model import WeightModel


class GeographicTemporalWeightedRandomForest(WeightModel, RegressorMixin):
    def __init__(self, local_estimator, *args, **kwargs):
        super().__init__(local_estimator, *args, **kwargs)
        self.geo_vector = None
        self.temporal_vector = None

    def show_residual(self, filename='residual'):
        return scatter_3d(
            self.geo_vector, self.temporal_vector, np.abs(self.local_residual_),
            'Residual Spatio-temporal Plot', 'Residual',
            filename
        )

    def show_y(self, filename='y'):
        return scatter_3d(
            self.geo_vector, self.temporal_vector, self.y,
            'Dependent Value Spatio-temporal distribution', 'Dependent Value',
            filename
        )

    def show_importance(self, permutation=False):
        if permutation:
            importance_array = self.importance_score_local()
            suffix = 'MDA'
        else:
            importance_array = self.importance_score_local()
            suffix = 'MDI'

        for index, feature_importance in enumerate(importance_array.T):
            scatter_3d(
                self.geo_vector, self.temporal_vector, feature_importance,
                f'Spatio-temporal Importance of Independent Variable {index} Using {suffix}',
                'Importance Value',
                f'importance_{index}_{suffix}'
            )
