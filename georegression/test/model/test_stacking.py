from time import time as t

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from georegression.stacking_model import StackingWeightModel
from georegression.test.data import load_HP
from georegression.weight_model import WeightModel

# X, y_true, xy_vector, time = load_TOD()
# X, y_true, xy_vector, time = load_ESI()
X, y_true, xy_vector, time = load_HP()


def test_stacking():
    local_estimator = DecisionTreeRegressor(splitter="random", max_depth=2)
    distance_measure = "euclidean"
    kernel_type = "bisquare"
    distance_ratio = None
    bandwidth = None
    neighbour_count = 0.01

    model = StackingWeightModel(
        local_estimator,
        distance_measure,
        kernel_type,
        distance_ratio,
        bandwidth,
        neighbour_count,
    )

    model.fit(X, y_true, [xy_vector, time])
    print(f"{model.llocv_score_}, {model.llocv_stacking_}")


def test_alpha():
    local_estimator = DecisionTreeRegressor(splitter="random", max_depth=2)
    distance_measure = "euclidean"
    kernel_type = "bisquare"
    distance_ratio = None
    bandwidth = None
    neighbour_count = 0.01

    model = StackingWeightModel(
        local_estimator,
        distance_measure,
        kernel_type,
        distance_ratio,
        bandwidth,
        neighbour_count,
        alpha=10,
    )

    model.fit(X, y_true, [xy_vector, time])
    print(f"{model.llocv_score_}, {model.llocv_stacking_}")

    for local_estimator in model.local_estimator_list:
        print(local_estimator.meta_estimator.coef_)
        break

    # For alpha=0.1, stacking_score = 0.5750569627981988
    """
    Coefficients of first stacking estimator:
    [-0.46379083 -0.38453714  0.39963185  0.01484807  0.16410479 -0.59694787
      0.21276714  0.11330034  0.29212005 -0.20581994  0.07942222  0.92542167
      0.44300962  0.26067723  0.03980381 -0.32809317  0.17886772  0.26176183
      0.31227637  0.12423833  0.23946592]
    """

    # For alpha=10, stacking_score = 0.9403979818713938
    """
    Coefficients of first stacking estimator:
    [ 0.07789433 -0.03072463  0.18275214 -0.00193438  0.05766076 -0.00123777
      0.13473063  0.1755927   0.0568057   0.0234573   0.14681941  0.03860493
     -0.06496593  0.1208457   0.06717717  0.0523331   0.0167307   0.14635798
     -0.03296376 -0.04416956  0.26379955]
    """


def test_estimator_sample():
    local_estimator = DecisionTreeRegressor(splitter="random", max_depth=2)
    distance_measure = "euclidean"
    kernel_type = "bisquare"
    distance_ratio = None
    bandwidth = None
    neighbour_count = 0.1

    model = StackingWeightModel(
        local_estimator,
        distance_measure,
        kernel_type,
        distance_ratio,
        bandwidth,
        neighbour_count,
        estimator_sample_rate=0.1,
    )

    model.fit(X, y_true, [xy_vector, time])
    print(f"{model.llocv_score_}, {model.llocv_stacking_}")

    model = StackingWeightModel(
        local_estimator,
        distance_measure,
        kernel_type,
        distance_ratio,
        bandwidth,
        neighbour_count,
        estimator_sample_rate=0.5,
    )
    model.fit(X, y_true, [xy_vector, time])
    print(f"{model.llocv_score_}, {model.llocv_stacking_}")

    model = StackingWeightModel(
        local_estimator,
        distance_measure,
        kernel_type,
        distance_ratio,
        bandwidth,
        neighbour_count,
        estimator_sample_rate=None,
    )
    model.fit(X, y_true, [xy_vector, time])
    print(f"{model.llocv_score_}, {model.llocv_stacking_}")


def test_performance():
    distance_measure = "euclidean"
    kernel_type = "bisquare"
    distance_ratio = None
    bandwidth = None
    neighbour_count = 0.1

    estimator = StackingWeightModel(
        ExtraTreeRegressor(splitter="random", max_depth=1),
        distance_measure,
        kernel_type,
        distance_ratio,
        bandwidth,
        neighbour_count,
        neighbour_leave_out_rate=0.1,
        # estimator_sample_rate=0.1,
    )

    t1 = t()
    estimator.fit(X, y_true, [xy_vector, time])
    t2 = t()
    print(t2 - t1, estimator.llocv_score_, estimator.llocv_stacking_)
    # neighbour_count = 0.1 neighbour_leave_out_rate=0.1 17.498192310333252 0.7670304978651743 0.8134413795992002

    estimator = WeightModel(
        RandomForestRegressor(n_estimators=50),
        distance_measure,
        kernel_type,
        distance_ratio,
        bandwidth,
        neighbour_count,
    )
    t2 = t()
    estimator.fit(X, y_true, [xy_vector, time])
    t3 = t()

    print(t3 - t2, estimator.llocv_score_)
    # neighbour_count = 0.1 n_estimators=50 34.99542546272278 0.8096408618045396

    estimator = WeightModel(
        RidgeCV(),
        distance_measure,
        kernel_type,
        distance_ratio,
        bandwidth,
        neighbour_count,
    )
    estimator.fit(X, y_true, [xy_vector, time])
    t4 = t()
    # neighbour_count = 0.1 5.488587141036987 0.7706704632683226

    print(t4 - t3, estimator.llocv_score_)


def test_stacking_not_leaking():
    # local_estimator = DecisionTreeRegressor(splitter="random", max_depth=2)
    # local_estimator = DecisionTreeRegressor()
    local_estimator = RandomForestRegressor(n_estimators=50)
    # local_estimator = RidgeCV()
    distance_measure = "euclidean"
    kernel_type = "bisquare"
    distance_ratio = None
    bandwidth = None

    def fit_wrapper():
        estimator = StackingWeightModel(
            local_estimator,
            distance_measure,
            kernel_type,
            distance_ratio,
            bandwidth,
            neighbour_count,
            neighbour_leave_out_rate=leave_out_rate,
        )

        t1 = t()
        estimator.fit(X, y_true, [xy_vector, time])
        t2 = t()
        print("neighbour_count =", neighbour_count, "leave_out_rate =", leave_out_rate)
        print(
            "time =",
            t2 - t1,
            "llocv_score =",
            estimator.llocv_score_,
            "llocv_stacking =",
            estimator.llocv_stacking_,
        )

    neighbour_count = 0.1
    leave_out_rate = 0.1

    fit_wrapper()

    estimator = WeightModel(
        RandomForestRegressor(n_estimators=50),
        distance_measure,
        kernel_type,
        distance_ratio,
        bandwidth,
        neighbour_count,
    )
    t2 = t()
    estimator.fit(X, y_true, [xy_vector, time])
    t3 = t()
    print(t3 - t2, estimator.llocv_score_)

    for depth in range(1, 31, 2):
        # local_estimator = DecisionTreeRegressor(max_depth=depth)
        # fit_wrapper()
        pass

    for neighbour_count in np.arange(0.05, 0.35, 0.05):
        for leave_out_rate in np.arange(0.05, 0.35, 0.05):
            # fit_wrapper()
            pass


if __name__ == "__main__":
    # test_stacking()
    # test_alpha()
    # test_estimator_sample()
    test_performance()
    # test_stacking_not_leaking()
