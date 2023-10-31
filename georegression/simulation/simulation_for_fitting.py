import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from georegression.simulation.simulation_utils import polynomial_function, sigmoid_function, sample_points, sample_x

from georegression.stacking_model import StackingWeightModel
from georegression.simulation.simulation import coef_strong
from georegression.weight_model import WeightModel


# TODO: Explain why the improvement becomes significant when there are more points.

def f_square(X, C, points):
    return (
            polynomial_function(C[0], 2)(X[:, 0], points) +
            C[0](points) * 10 +
            0
    )

def f_sigmoid(X, C, points):
    return (
            sigmoid_function(C[0])(X[:, 0], points) +
            0
    )


def generate_sample(count, f, coef_func, random_seed=1):
    np.random.seed(random_seed)
    points = sample_points(count)
    x1 = sample_x(count)
    coefficients = [coef_func()]

    X = np.stack((x1, ), axis=-1)
    y = f(X, coefficients, points)

    return X, y, points



def square_strong():
    X, y, points = generate_sample(5000, f_square, coef_strong, random_seed=1)
    X_plus = np.concatenate([X, points], axis=1)

    distance_measure = "euclidean"
    kernel_type = "bisquare"

    model = StackingWeightModel(
        DecisionTreeRegressor(splitter="random", max_depth=1),
        distance_measure,
        kernel_type,
        neighbour_count=0.03,
        neighbour_leave_out_rate=0.15,
    )
    model.fit(X_plus, y, [points])
    print('Stacking:', model.llocv_score_, model.llocv_stacking_)

    model = WeightModel(
        RandomForestRegressor(n_estimators=50),
        distance_measure,
        kernel_type,
        neighbour_count=0.03,
    )
    model.fit(X_plus, y, [points])
    print('GRF:', model.llocv_score_)

    model = WeightModel(
        LinearRegression(),
        distance_measure,
        kernel_type,
        neighbour_count=0.03,
    )
    model.fit(X_plus, y, [points])
    print('GWR:', model.llocv_score_)

    model = RandomForestRegressor(oob_score=True, n_estimators=2000, n_jobs=-1)
    model.fit(X_plus, y)
    print('RF:', model.oob_score_)

    model = LinearRegression()
    model.fit(X_plus, y)
    print('LR:', model.score(X_plus, y))


def test_GRF():
    distance_measure = "euclidean"
    kernel_type = "bisquare"

    for neighbour_count in [0.01, 0.02, 0.05]:
        model = WeightModel(
            RandomForestRegressor(n_estimators=50),
            distance_measure,
            kernel_type,
            neighbour_count=neighbour_count,
        )
        model.fit(X, y, [points])
        print('GRF:', model.llocv_score_, neighbour_count)


def test_stacking():
    distance_measure = "euclidean"
    kernel_type = "bisquare"
    local_estimator = DecisionTreeRegressor(splitter="random", max_depth=1)

    for neighbour_count in [0.008, 0.01, 0.012]:
        for leave_out_rate in [0.15, 0.25]:
            model = StackingWeightModel(
                local_estimator,
                distance_measure,
                kernel_type,
                neighbour_count=neighbour_count,
                neighbour_leave_out_rate=leave_out_rate,
            )
            model.fit(X, y, [points])
            print('Stacking:', model.llocv_score_,
                  model.llocv_stacking_, neighbour_count, leave_out_rate)

