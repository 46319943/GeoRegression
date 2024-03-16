import json
import os
import time
from functools import partial

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

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
    info=None,
):
    if info is None:
        info = {}

    X_plus = np.concatenate([X, points], axis=1)

    distance_measure = "euclidean"
    kernel_type = "bisquare"

    result = {}

    model = StackingWeightModel(
        DecisionTreeRegressor(splitter="random", max_depth=X.shape[1]),
        distance_measure,
        kernel_type,
        neighbour_count=stacking_neighbour_count,
        neighbour_leave_out_rate=stacking_neighbour_leave_out_rate,
    )

    repeats = 10
    stackings = []
    for i in range(repeats):
        t1 = time.time()
        model.fit(X_plus, y, [points])
        t2 = time.time()

        stackings.append(model.llocv_stacking_)

        print("Stacking:", model.llocv_score_, model.llocv_stacking_)
        print(t2 - t1)

        result[f"Stacking_Base_{i}"] = model.llocv_score_
        result[f"Stacking_{i}"] = model.llocv_stacking_
        result[f"Stacking_Time"] = t2 - t1

    result["Stacking_Mean"] = np.mean(stackings)
    result["Stacking_Std"] = np.std(stackings)
    result["Stacking_Max"] = np.max(stackings)
    result["Stacking_Min"] = np.min(stackings)

    model = StackingWeightModel(
        ExtraTreesRegressor(n_estimators=10, max_depth=X.shape[1]),
        distance_measure,
        kernel_type,
        neighbour_count=stacking_neighbour_count,
        neighbour_leave_out_rate=stacking_neighbour_leave_out_rate,
    )

    stackings_extra = []
    for i in range(repeats):
        t1 = time.time()
        model.fit(X_plus, y, [points])
        t2 = time.time()

        stackings_extra.append(model.llocv_stacking_)

        print("Stacking_Extra:", model.llocv_score_, model.llocv_stacking_)
        print(t2 - t1)

        result[f"Stacking_Extra_Base_{i}"] = model.llocv_score_
        result[f"Stacking_Extra_{i}"] = model.llocv_stacking_
        result[f"Stacking_Extra_Time"] = t2 - t1

    result["Stacking_Extra_Mean"] = np.mean(stackings_extra)
    result["Stacking_Extra_Std"] = np.std(stackings_extra)
    result["Stacking_Extra_Max"] = np.max(stackings_extra)
    result["Stacking_Extra_Min"] = np.min(stackings_extra)

    result = {**result, **info}
    with open("simulation_variation.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")

    return result


def coef_auto_gau_strong():
    coef_radial = radial_coefficient(np.array([0, 0]), 1 / np.sqrt(200))
    coef_dir = directional_coefficient(np.array([1, 1]))

    gau_coef_list = []
    for _ in range(1000):
        # Randomly generate the parameters for gaussian coefficient
        center = np.random.uniform(-10, 10, 2)
        amplitude = np.random.uniform(1, 2)
        sign = np.random.choice([-1, 1])
        amplitude *= sign
        sigma1 = np.random.uniform(0.2, 1)
        sigma2 = np.random.uniform(0.2, 1)
        cov = np.random.uniform(-np.sqrt(sigma1 * sigma2), np.sqrt(sigma1 * sigma2))
        sigma = np.array([[sigma1, cov], [cov, sigma2]])

        coef_gau = gaussian_coefficient(center, sigma, amplitude=amplitude)
        gau_coef_list.append(coef_gau)

    coef_gau = coefficient_wrapper(np.sum, *gau_coef_list)
    coef_sum = coefficient_wrapper(np.sum, coef_radial, coef_dir, coef_gau)

    return coef_sum


def f_square(X, C, points):
    return polynomial_function(C[0], 2)(X[:, 0], points) + 0


def generate_sample(count, f, coef_func, random_seed=1, plot=False):
    np.random.seed(random_seed)
    points = sample_points(count, bounds=(-10, 10))
    x1 = sample_x(count, bounds=(-10, 10))
    coefficients = [coef_func()]

    X = np.stack((x1,), axis=-1)
    y = f(X, coefficients, points)

    if plot:
        folder = f"Plot/{coef_func.__name__}_{f.__name__}_{count}"
        os.makedirs(folder, exist_ok=True)
        show_sample(X, y, points, coefficients, folder)

    return X, y, points


def square_gau_strong_100():
    X, y, points = generate_sample(
        100, f_square, coef_auto_gau_strong, random_seed=1, plot=True
    )

    return fit_models(
        X,
        y,
        points,
        stacking_neighbour_count=0.45,
        stacking_neighbour_leave_out_rate=0.2,
        info={"f": "f_square", "coef": "coef_gau_strong", "count": 100},
    )


def square_gau_strong_500():
    X, y, points = generate_sample(
        500, f_square, coef_auto_gau_strong, random_seed=1, plot=True
    )

    return fit_models(
        X,
        y,
        points,
        stacking_neighbour_count=0.08,
        stacking_neighbour_leave_out_rate=0.1,
        info={"f": "f_square", "coef": "coef_gau_strong", "count": 500},
    )


def square_gau_strong_1000():
    X, y, points = generate_sample(
        1000, f_square, coef_auto_gau_strong, random_seed=1, plot=True
    )

    return fit_models(
        X,
        y,
        points,
        stacking_neighbour_count=0.02,
        stacking_neighbour_leave_out_rate=0.05,
        info={"f": "f_square", "coef": "coef_gau_strong", "count": 1000},
    )


def square_gau_strong_5000():
    X, y, points = generate_sample(
        5000, f_square, coef_auto_gau_strong, random_seed=1, plot=True
    )

    return fit_models(
        X,
        y,
        points,
        stacking_neighbour_count=0.008,
        stacking_neighbour_leave_out_rate=0.2,
        info={"f": "f_square", "coef": "coef_gau_strong", "count": 5000},
    )


def square_gau_strong_10000():
    X, y, points = generate_sample(
        10000, f_square, coef_auto_gau_strong, random_seed=1, plot=True
    )

    return fit_models(
        X,
        y,
        points,
        stacking_neighbour_count=0.008,
        stacking_neighbour_leave_out_rate=0.2,
        info={"f": "f_square", "coef": "coef_gau_strong", "count": 10000},
    )





if __name__ == "__main__":
    # square_gau_strong_100()
    # square_gau_strong_500()
    square_gau_strong_1000()
    # square_gau_strong_5000()
    # square_gau_strong_10000()

    pass
