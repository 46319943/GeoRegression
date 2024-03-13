import json
import time

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from georegression.simulation.simulation import show_sample
from georegression.simulation.simulation_utils import *
from georegression.stacking_model import StackingWeightModel
from georegression.weight_model import WeightModel


# TODO: Explain why the improvement becomes significant when there are more points.


def fit_models(
    X,
    y,
    points,
    stacking_neighbour_count=0.03,
    stacking_neighbour_leave_out_rate=0.15,
    grf_neighbour_count=0.03,
    grf_n_estimators=50,
    gwr_neighbour_count=0.03,
    rf_n_estimators=2000,
):
    X_plus = np.concatenate([X, points], axis=1)

    distance_measure = "euclidean"
    kernel_type = "bisquare"

    result = {}

    model = StackingWeightModel(
        DecisionTreeRegressor(splitter="random", max_depth=1),
        distance_measure,
        kernel_type,
        neighbour_count=stacking_neighbour_count,
        neighbour_leave_out_rate=stacking_neighbour_leave_out_rate,
    )
    t1 = time.time()
    model.fit(X_plus, y, [points])
    t2 = time.time()
    print("Stacking:", model.llocv_score_, model.llocv_stacking_)
    print(t2 - t1)
    result["Stacking_Base"] = model.llocv_score_
    result["Stacking"] = model.llocv_stacking_
    result["Stacking_Time"] = t2 - t1

    model = WeightModel(
        RandomForestRegressor(n_estimators=grf_n_estimators),
        distance_measure,
        kernel_type,
        neighbour_count=grf_neighbour_count,
    )
    t1 = time.time()
    model.fit(X_plus, y, [points])
    t2 = time.time()
    print("GRF:", model.llocv_score_)
    print(t2 - t1)
    result["GRF"] = model.llocv_score_
    result["GRF_Time"] = t2 - t1

    model = WeightModel(
        LinearRegression(),
        distance_measure,
        kernel_type,
        neighbour_count=gwr_neighbour_count,
    )
    t1 = time.time()
    model.fit(X_plus, y, [points])
    t2 = time.time()
    print("GWR:", model.llocv_score_)
    print(t2 - t1)
    result["GWR"] = model.llocv_score_
    result["GWR_Time"] = t2 - t1

    model = RandomForestRegressor(
        oob_score=True, n_estimators=rf_n_estimators, n_jobs=-1
    )
    t1 = time.time()
    model.fit(X_plus, y)
    t2 = time.time()
    print("RF:", model.oob_score_)
    print(t2 - t1)
    result["RF"] = model.oob_score_
    result["RF_Time"] = t2 - t1

    model = LinearRegression()
    t1 = time.time()
    model.fit(X_plus, y)
    t2 = time.time()
    print("LR:", model.score(X_plus, y))
    print(t2 - t1)
    result["LR"] = model.score(X_plus, y)
    result["LR_Time"] = t2 - t1

    with open("simulation_result.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")

    return result


def coef_auto_gau_weak():
    coef_radial = radial_coefficient(np.array([0, 0]), 1 / np.sqrt(200))
    coef_dir = directional_coefficient(np.array([1, 1]))

    gau_coef_list = []
    for i in range(1000):
        # Randomly generate the parameters for gaussian coefficient
        center = np.random.uniform(-10, 10, 2)
        amplitude = np.random.uniform(1, 2)
        sign = np.random.choice([-1, 1])
        amplitude *= sign
        sigma1 = np.random.uniform(0.5, 5)
        sigma2 = np.random.uniform(0.5, 5)
        cov = np.random.uniform(-np.sqrt(sigma1 * sigma2), np.sqrt(sigma1 * sigma2))
        sigma = np.array([[sigma1, cov], [cov, sigma2]])

        coef_gau = gaussian_coefficient(center, sigma, amplitude=amplitude)
        gau_coef_list.append(coef_gau)

    coef_gau = coefficient_wrapper(np.sum, *gau_coef_list)
    coef_sum = coefficient_wrapper(np.sum, coef_radial, coef_dir, coef_gau)

    return coef_sum


def coef_strong():
    coef_radial = radial_coefficient(np.array([0, 0]), 1 / np.sqrt(200))
    coef_dir = directional_coefficient(np.array([1, 1]))
    coef_sin_1 = sine_coefficient(1, np.array([-1, 1]), 1)
    coef_sin_2 = sine_coefficient(1, np.array([1, 1]), 1)
    coef_sin = coefficient_wrapper(np.sum, coef_sin_1, coef_sin_2)
    coef_gau_1 = gaussian_coefficient(np.array([-5, 5]), 3)
    coef_gau_2 = gaussian_coefficient(np.array([-5, -5]), 3, amplitude=2)
    coef_gau = coefficient_wrapper(np.sum, coef_gau_1, coef_gau_2)

    coef_sum = coefficient_wrapper(np.sum, coef_radial, coef_dir, coef_sin, coef_gau)

    return coef_sum


def f_square(X, C, points):
    return (
        polynomial_function(C[0], 2)(X[:, 0], points)
        # + C[0](points) * 10
        + 0
    )


def f_sigmoid(X, C, points):
    return sigmoid_function(C[0])(X[:, 0], points) + 0


def generate_sample(count, f, coef_func, random_seed=1, plot=False):
    np.random.seed(random_seed)
    points = sample_points(count, bounds=(-10, 10))
    x1 = sample_x(count, bounds=(-10, 10))
    coefficients = [coef_func()]

    X = np.stack((x1,), axis=-1)
    y = f(X, coefficients, points)

    if plot:
        show_sample(X, y, points, coefficients)

    return X, y, points


def square_strong_100():
    X, y, points = generate_sample(100, f_square, coef_strong, random_seed=1, plot=True)
    # test_models(
    #     X,
    #     y,
    #     points,
    #     [0.1, 0.2, 0.3],
    #     [0.1, 0.2, 0.3],
    #     [0.15, 0.3],
    #     100,
    #     "f_square",
    #     "coef_strong",
    # )

    fit_models(
        X,
        y,
        points,
        stacking_neighbour_count=0.3,
        stacking_neighbour_leave_out_rate=0.3,
        grf_neighbour_count=0.2,
        grf_n_estimators=50,
        gwr_neighbour_count=0.2,
        rf_n_estimators=2000,
    )


def square_strong_500():
    X, y, points = generate_sample(500, f_square, coef_strong, random_seed=1, plot=True)
    # test_models(
    #     X,
    #     y,
    #     points,
    #     [0.1, 0.2, 0.3],
    #     [0.1, 0.2, 0.3],
    #     [0.15, 0.3],
    #     500,
    #     "f_square",
    #     "coef_strong",
    # )

    fit_models(
        X,
        y,
        points,
        stacking_neighbour_count=0.2,
        stacking_neighbour_leave_out_rate=0.15,
        grf_neighbour_count=0.2,
        grf_n_estimators=50,
        gwr_neighbour_count=0.2,
        rf_n_estimators=2000,
    )


def square_strong_5000():
    X, y, points = generate_sample(
        5000, f_square, coef_strong, random_seed=1, plot=True
    )
    test_models(
        X,
        y,
        points,
        [0.02, 0.03, 0.04, 0.05],
        [0.1, 0.2, 0.3],
        [0.02, 0.03, 0.04, 0.05],
        5000,
        "f_square",
        "coef_strong",
    )

    # fit_models(
    #     X,
    #     y,
    #     points,
    #     stacking_neighbour_count=0.03,
    #     stacking_neighbour_leave_out_rate=0.15,
    #     grf_neighbour_count=0.03,
    #     grf_n_estimators=50,
    #     gwr_neighbour_count=0.03,
    #     rf_n_estimators=2000,
    # )


def square_gau_strong():
    # X, y, points = generate_sample(5000, f_square, coef_auto_gau_strong, random_seed=1, plot=True)
    # X, y, points = generate_sample(5000, f_sigmoid, coef_strong, random_seed=1, plot=False)
    pass


def test_models(
    X,
    y,
    points,
    stacking_neighbour_count,
    stacking_neighbour_leave_out_rate,
    grf_neighbour_count,
    count,
    func,
    coef,
):
    stacking_params = test_stacking(
        X, y, points, stacking_neighbour_count, stacking_neighbour_leave_out_rate
    )
    grf_params = test_GRF(X, y, points, grf_neighbour_count)

    with open("simulation_params.jsonl", "a") as f:
        for params in stacking_params:
            params["count"] = count
            params["func"] = func
            params["coef"] = coef
            f.write(json.dumps(params) + "\n")
        for params in grf_params:
            params["count"] = count
            params["func"] = func
            params["coef"] = coef
            f.write(json.dumps(params) + "\n")

    # Print the param with the best score
    print(max(stacking_params, key=lambda x: x["Stacking"]))
    print(max(grf_params, key=lambda x: x["GRF"]))


def test_GRF(X, y, points, neighbour_counts):
    X_plus = np.concatenate([X, points], axis=1)

    distance_measure = "euclidean"
    kernel_type = "bisquare"

    result = []

    for neighbour_count in neighbour_counts:
        model = WeightModel(
            RandomForestRegressor(n_estimators=50),
            distance_measure,
            kernel_type,
            neighbour_count=neighbour_count,
        )
        model.fit(X_plus, y, [points])
        print("GRF:", model.llocv_score_, neighbour_count)
        result.append({"GRF": model.llocv_score_, "neighbour_count": neighbour_count})

    return result


def test_stacking(X, y, points, neighbour_counts, leave_out_rates):
    distance_measure = "euclidean"
    kernel_type = "bisquare"
    local_estimator = DecisionTreeRegressor(splitter="random", max_depth=1)

    result = []

    for neighbour_count in neighbour_counts:
        for leave_out_rate in leave_out_rates:
            model = StackingWeightModel(
                local_estimator,
                distance_measure,
                kernel_type,
                neighbour_count=neighbour_count,
                neighbour_leave_out_rate=leave_out_rate,
            )
            model.fit(X, y, [points])
            print(
                "Stacking:",
                model.llocv_score_,
                model.llocv_stacking_,
                neighbour_count,
                leave_out_rate,
            )
            result.append(
                {
                    "Stacking_Base": model.llocv_score_,
                    "Stacking": model.llocv_stacking_,
                    "neighbour_count": neighbour_count,
                    "leave_out_rate": leave_out_rate,
                }
            )

    return result


if __name__ == "__main__":
    # square_strong_100()
    # square_strong_500()
    square_strong_5000()
