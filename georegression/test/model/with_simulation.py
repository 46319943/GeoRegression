from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from georegression.stacking_model import StackingWeightModel
from georegression.test.data.simulation import generate_sample
from georegression.weight_model import WeightModel

X, y, points, coefficients = generate_sample()


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
    X, y, points, coefficients = generate_sample(count=500)

    local_estimator = DecisionTreeRegressor(splitter="random", max_depth=2)
    distance_measure = "euclidean"
    kernel_type = "bisquare"
    neighbour_count = 0.4

    model = StackingWeightModel(
        local_estimator,
        distance_measure,
        kernel_type,
        neighbour_count=neighbour_count,
        neighbour_leave_out_rate=0.1,
    )

    model.fit(X, y, [points])
    print(model.llocv_stacking_)

    model = WeightModel(
        RandomForestRegressor(),
        distance_measure,
        kernel_type,
        neighbour_count=neighbour_count,
    )

    model.fit(X, y, [points])
    print(model.llocv_score_)

    model = RandomForestRegressor(oob_score=True)
    model.fit(X, y)
    print(model.oob_score_)

    """
    0.7813469046663418
    0.7334095600009363
    0.5015610109930759
    """


if __name__ == "__main__":
    # test_nonlinear_spatiotemporal_really_work()
    test_robust_under_various_data()
