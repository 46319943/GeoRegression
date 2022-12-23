from sklearn.base import RegressorMixin

from georegression.weight_model import WeightModel


class GeographicTemporalWeightedRandomForest(WeightModel, RegressorMixin):
    def __init__(self, local_estimator, *args, **kwargs):
        super().__init__(local_estimator, *args, **kwargs)
        self.geo_vector = None
        self.temporal_vector = None
