import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from georegression.stacking_model import StackingWeightModel
from georegression.test.data.simulation import generate_sample
from georegression.weight_model import WeightModel

X, y, points, _ = generate_sample()


def test_nonlinear_spatiotemporal_really_work():
    neighbour_count = 0.4

    model = WeightModel(
        LinearRegression(),
        distance_measure="euclidean",
        kernel_type="bisquare",
        neighbour_count=neighbour_count,
        cache_data=True,
        cache_estimator=True,
    )

    model.fit(X, y, [points])
    print(model.llocv_score_)

    model = WeightModel(
        RandomForestRegressor(n_estimators=50),
        distance_measure="euclidean",
        kernel_type="bisquare",
        neighbour_count=neighbour_count,
        cache_data=True,
        cache_estimator=True,
    )

    model.fit(X, y, [points])
    print(model.llocv_score_)

    # Fit random forest on the data and print oob score
    model = RandomForestRegressor(oob_score=True)
    model.fit(X, y)
    print(model.oob_score_)


def test_robust_under_various_data():
    """
    TODO: Explain why the improvement becomes significant when there are more points.

    Returns:

    """
    X, y, points, _ = generate_sample(count=1000, random_seed=1)
    X_plus = np.concatenate([X, points], axis=1)

    local_estimator = DecisionTreeRegressor(splitter="random", max_depth=1)
    distance_measure = "euclidean"
    kernel_type = "bisquare"

    neighbour_count = 0.03

    model = StackingWeightModel(
        local_estimator,
        distance_measure,
        kernel_type,
        neighbour_count=neighbour_count,
        neighbour_leave_out_rate=0.15,
    )
    model.fit(X_plus, y, [points])
    print('Stacking:', model.llocv_score_, model.llocv_stacking_)

    model = WeightModel(
        RandomForestRegressor(n_estimators=200),
        distance_measure,
        kernel_type,
        neighbour_count=neighbour_count,
    )
    model.fit(X_plus, y, [points])
    print('GRF:', model.llocv_score_)

    model = WeightModel(
        LinearRegression(),
        distance_measure,
        kernel_type,
        neighbour_count=neighbour_count,
    )
    model.fit(X_plus, y, [points])
    print('GWR:', model.llocv_score_)

    model = RandomForestRegressor(oob_score=True, n_estimators=5500, n_jobs=-1)
    model.fit(X_plus, y)
    print('RF:', model.oob_score_)

    model = LinearRegression()
    model.fit(X_plus, y)
    print('LR:', model.score(X_plus, y))

    """
    count=500
    0.7813469046663418
    0.7334095600009363
    0.5015610109930759
    
    count=5000
    0.8692464077285508
    0.7648331307574766
    0.5087431918278111
    
    count=5000
    neighbour_count = 0.1
    0.94041696882913
    0.9466926082507984
    0.5182052654589111
    """


def test_without_X_plus():
    X, y, points, _ = generate_sample(count=1000, random_seed=1)
    X_plus = np.concatenate([X, points], axis=1)

    local_estimator = DecisionTreeRegressor(splitter="random", max_depth=1)
    distance_measure = "euclidean"
    kernel_type = "bisquare"

    neighbour_count = 0.03

    model = StackingWeightModel(
        local_estimator,
        distance_measure,
        kernel_type,
        neighbour_count=neighbour_count,
        neighbour_leave_out_rate=0.15,
    )
    model.fit(X, y, [points])
    print('Stacking:', model.llocv_score_, model.llocv_stacking_)

    model = WeightModel(
        RandomForestRegressor(n_estimators=200),
        distance_measure,
        kernel_type,
        neighbour_count=neighbour_count,
    )
    model.fit(X, y, [points])
    print('GRF:', model.llocv_score_)

    model = WeightModel(
        LinearRegression(),
        distance_measure,
        kernel_type,
        neighbour_count=neighbour_count,
    )
    model.fit(X, y, [points])
    print('GWR:', model.llocv_score_)

    model = RandomForestRegressor(oob_score=True, n_estimators=5500, n_jobs=-1)
    model.fit(X_plus, y)
    print('RF:', model.oob_score_)

    model = LinearRegression()
    model.fit(X_plus, y)
    print('LR:', model.score(X_plus, y))

def draw_graph():
    pass


if __name__ == "__main__":
    # test_nonlinear_spatiotemporal_really_work()
    # test_robust_under_various_data()

    test_without_X_plus()