from time import time as t

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.tree import DecisionTreeRegressor

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
    midpoint = True
    p = None

    model = StackingWeightModel(
        local_estimator,
        distance_measure,
        kernel_type,
        distance_ratio,
        bandwidth,
        neighbour_count,
        midpoint,
        p,
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
    midpoint = True
    p = None

    model = StackingWeightModel(
        local_estimator,
        distance_measure,
        kernel_type,
        distance_ratio,
        bandwidth,
        neighbour_count,
        midpoint,
        p,
        alpha=10,
    )

    model.fit(X, y_true, [xy_vector, time])
    print(f"{model.llocv_score_}, {model.llocv_stacking_}")

    for local_estimator in model.local_estimator_list:
        print(local_estimator.final_estimator.coef_)
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
    midpoint = True
    p = None

    model = StackingWeightModel(
        local_estimator,
        distance_measure,
        kernel_type,
        distance_ratio,
        bandwidth,
        neighbour_count,
        midpoint,
        p,
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
        midpoint,
        p,
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
        midpoint,
        p,
        estimator_sample_rate=None,
    )
    model.fit(X, y_true, [xy_vector, time])
    print(f"{model.llocv_score_}, {model.llocv_stacking_}")


def test_performance():
    local_estimator = DecisionTreeRegressor(splitter="random", max_depth=2)
    distance_measure = "euclidean"
    kernel_type = "bisquare"
    distance_ratio = None
    bandwidth = None
    neighbour_count = 0.01
    midpoint = True
    p = None

    estimator = StackingWeightModel(
        local_estimator,
        distance_measure,
        kernel_type,
        distance_ratio,
        bandwidth,
        neighbour_count,
        midpoint,
        p,
        neighbour_leave_out_rate=0.1,
        # estimator_sample_rate=0.1,
    )

    t1 = t()
    estimator.fit(X, y_true, [xy_vector, time])
    t2 = t()
    print(t2 - t1, estimator.llocv_score_, estimator.llocv_stacking_)
    # 917.4381189346313 0.7218072208798815 0.8274797786820569
    # 185.95322585105896 0.7314718925728154 0.8160008434522774
    # 471.748841047287 0.7170324025502195 0.8186682469210496
    # neighbour_count = 0.01 12.433058261871338 0.7264394281767108 0.9421304744738035
    # neighbour_count = 0.1 59.23662233352661 0.7549265112954625 0.8533792294300071
    # neighbour_count = 0.1 estimator_sample_rate=0.1 17.0358464717865 0.7530776263728781 0.8559844494270573
    # neighbour_count = 0.01 neighbour_leave_out_rate=0.1 9.614805459976196 0.7025745058476021 0.8849390576510993
    # neighbour_count = 0.1 neighbour_leave_out_rate=0.1 15.770824909210205 0.7680928707118291 0.8556669876852211
    # neighbour_count = 0.1 neighbour_leave_out_rate=0.1 14.831573724746704 0.7644013832778737 0.8101877198579331
    # neighbour_count = 0.01 neighbour_leave_out_rate=0.1 9.51764440536499 0.7088130603052478 0.7641255806160333

    estimator = WeightModel(
        RandomForestRegressor(n_estimators=50),
        distance_measure,
        kernel_type,
        distance_ratio,
        bandwidth,
        neighbour_count,
        midpoint,
        p,
    )
    estimator.fit(X, y_true, [xy_vector])
    t3 = t()

    print(t3 - t2, estimator.llocv_score_)
    # 1075.0753190517426 0.8228107533011373
    # 1071.8284921646118 0.8242928357531293
    # neighbour_count = 0.01 28.833017110824585 0.8278781002714682
    # neighbour_count = 0.1 183.82665371894836 0.8354290564175364
    # neighbour_count = 0.01 n_estimators=5  3.1212289333343506 0.7988055718383715
    # neighbour_count = 0.1 n_estimators=5  10.742109775543213 0.7971370484169908
    # neighbour_count = 0.1 n_estimators=50  87.76663708686829 0.8333164789667881
    # neighbour_count = 0.01 n_estimators=50  87.76663708686829 0.8333164789667881

    estimator = WeightModel(
        RidgeCV(),
        distance_measure,
        kernel_type,
        distance_ratio,
        bandwidth,
        neighbour_count,
        midpoint,
        p,
    )
    # estimator.local_estimator = RidgeCV()
    # estimator.use_stacking = False
    estimator.fit(X, y_true, [xy_vector])
    t4 = t()

    print(t4 - t3, estimator.llocv_score_)
    # neighbour_count = 0.01 2.439164876937866 -2355843.4163171016
    # neighbour_count = 0.1 4.386993885040283 0.7985700705302594
    # neighbour_count = 0.01 RidgeCV 2.1749119758605957 0.736926407604787
    # neighbour_count = 0.1 RidgeCV 4.276575803756714 0.800494195041368
    # neighbour_count = 0.01 RidgeCV 2.18503475189209 0.736926407604787


def test_stacking_not_leaking():
    # local_estimator = DecisionTreeRegressor(splitter="random", max_depth=5)
    local_estimator = DecisionTreeRegressor(max_depth=3)
    # local_estimator = RidgeCV()
    distance_measure = "euclidean"
    kernel_type = "bisquare"
    distance_ratio = None
    bandwidth = None
    midpoint = True
    p = None

    def fit_wrapper():
        estimator = StackingWeightModel(
            local_estimator,
            distance_measure,
            kernel_type,
            distance_ratio,
            bandwidth,
            neighbour_count,
            midpoint,
            p,
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

    # fit_wrapper()

    for depth in range(1, 10):
        local_estimator = DecisionTreeRegressor(max_depth=depth)
        fit_wrapper()

    for neighbour_count in np.arange(0.05, 0.35, 0.05):
        for leave_out_rate in np.arange(0.05, 0.35, 0.05):
            pass
            # fit_wrapper()


if __name__ == "__main__":
    # test_stacking()
    # test_alpha()
    # test_estimator_sample()
    # test_performance()
    test_stacking_not_leaking()
