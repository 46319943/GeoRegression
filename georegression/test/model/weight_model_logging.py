from time import time as t

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from georegression.stacking_model import StackingWeightModel
from georegression.test.data import load_HP
from georegression.weight_model import WeightModel

X, y_true, xy_vector, time = load_HP()


def test_logging():
    distance_measure = "euclidean"
    kernel_type = "bisquare"
    distance_ratio = None
    bandwidth = None
    neighbour_count = 0.01

    estimator = WeightModel(
        RandomForestRegressor(n_estimators=25),
        distance_measure,
        kernel_type,
        distance_ratio,
        bandwidth,
        neighbour_count,
    )

    estimator.fit(X, y_true, [xy_vector, time])

if __name__ == '__main__':
    test_logging()