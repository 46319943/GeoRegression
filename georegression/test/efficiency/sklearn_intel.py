from sklearnex import patch_sklearn

# patch_sklearn()

from georegression.weight_model import WeightModel
from time import time as t

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from georegression.stacking_model import StackingWeightModel
from georegression.test.data import load_HP
from georegression.weight_model import WeightModel

X, y_true, xy_vector, time = load_HP()


def test_performance():
    distance_measure = "euclidean"
    kernel_type = "bisquare"
    neighbour_count = 0.1

    estimator = WeightModel(
        RandomForestRegressor(n_estimators=50, max_features=1.0),
        distance_measure,
        kernel_type,
        neighbour_count=neighbour_count,
    )
    t2 = t()
    estimator.fit(X, y_true, [xy_vector, time])
    t3 = t()
    print(f"Time taken to fit: {t3 - t2}")
    print(estimator.llocv_score_)
    # * neighbour_count = 0.01 n_estimators=50 12.85382342338562
    # * neighbour_count = 0.1 n_estimators=50 21.567977905273438 -1095426075.984387
    # neighbour_count = 0.01 n_estimators=50 19.778754949569702
    # neighbour_count = 0.1 n_estimators=50 38.774967432022095 0.8090267793183445
    # TODO: intel extension not working. Need to check the reason.


def test_performance_on_stacking():
    distance_measure = "euclidean"
    kernel_type = "bisquare"
    neighbour_count = 0.1

    estimator = StackingWeightModel(
        ExtraTreeRegressor(splitter="random", max_depth=1),
        distance_measure,
        kernel_type,
        neighbour_count=neighbour_count,
        neighbour_leave_out_rate=0.1,
    )


if __name__ == "__main__":
    test_performance()
