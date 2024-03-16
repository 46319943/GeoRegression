import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from georegression.local_ale import weighted_ale

from georegression.stacking_model import StackingWeightModel
from georegression.simulation.simulation import show_sample
from georegression.visualize.ale import plot_ale
from georegression.weight_model import WeightModel
from georegression.simulation.simulation_utils import *


def coef_manual_gau():
    coef_radial = radial_coefficient(np.array([0, 0]), 1 / np.sqrt(200))
    coef_dir = directional_coefficient(np.array([1, 1]))

    coef_gau_1 = gaussian_coefficient(np.array([-5, 5]), [[3, 4], [4, 8]], amplitude=-1)
    coef_gau_2 = gaussian_coefficient(np.array([-2, -5]), 5, amplitude=2)
    coef_gau_3 = gaussian_coefficient(np.array([8, 3]), 10, amplitude=-1.5)
    coef_gau_4 = gaussian_coefficient(
        np.array([2, 8]), [[3, 0], [0, 15]], amplitude=0.8
    )
    coef_gau_5 = gaussian_coefficient(np.array([5, -10]), 1, amplitude=1)
    coef_gau_6 = gaussian_coefficient(np.array([-10, -10]), 15, amplitude=1.5)
    coef_gau_6 = gaussian_coefficient(np.array([-11, 0]), 5, amplitude=2)
    coef_gau_6 = gaussian_coefficient(np.array([-11, 0]), 5, amplitude=2)
    coef_gau = coefficient_wrapper(
        np.sum, coef_gau_1, coef_gau_2, coef_gau_3, coef_gau_4, coef_gau_5, coef_gau_6
    )

    # coef_sum = coefficient_wrapper(np.sum, coef_radial, coef_dir, coef_gau)
    coef_sum = coefficient_wrapper(np.sum, coef_radial, coef_gau)

    return coef_sum


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

    coef_sum = coefficient_wrapper(partial(np.multiply, 2), coef_sum)

    return coef_sum


def f_square_2(X, C, points):
    return (
        polynomial_function(C[0], 2)(X[:, 0], points)
        + polynomial_function(C[1], 2)(X[:, 1], points)
        + 0
    )


def generate_sample(count, f, coef_func, random_seed=1, plot=False):
    np.random.seed(random_seed)
    points = sample_points(count, bounds=(-10, 10))
    x1 = sample_x(count, bounds=(-10, 10))
    x2 = sample_x(count, bounds=(-10, 10))

    if isinstance(coef_func, list):
        coefficients = [func() for func in coef_func]
    else:
        coefficients = [coef_func()]

    X = np.stack((x1, x2), axis=-1)
    y = f(X, coefficients, points)

    if plot:
        show_sample(X, y, points, coefficients)

    return X, y, points


def draw_graph():
    X, y, points = generate_sample(
        count=5000, f=f_square_2, coef_func=[coef_auto_gau_strong, coef_auto_gau_weak], random_seed=1,
        plot=True
    )
    X_plus = np.concatenate([X, points], axis=1)
    distance_measure = "euclidean"
    kernel_type = "bisquare"
    neighbour_count = 0.02

    model = WeightModel(
        RandomForestRegressor(n_estimators=50),
        distance_measure,
        kernel_type,
        neighbour_count=neighbour_count,
        cache_data=True,
        cache_estimator=True,
    )
    model.fit(X, y, [points])
    print("GRF:", model.llocv_score_)

    importance_global = model.importance_score_global()
    print(importance_global)

    importance_local = model.importance_score_local()
    print(importance_local)

    # Normalize the local importance to [0, 1]
    # importance_local = (importance_local - importance_local.min(axis=0)) / (
    #     importance_local.max(axis=0) - importance_local.min(axis=0)
    # )
    importance_local = (importance_local - importance_local.min(axis=1)) / (
        importance_local.max(axis=1) - importance_local.min(axis=1)
    )

    # Plot the local importance
    for i in range(importance_local.shape[1]):
        fig = plt.figure()
        scatter = plt.scatter(
            points[:, 0], points[:, 1], c=importance_local[:, i], cmap="viridis"
        )
        fig.colorbar(scatter)
        fig.savefig(f"Plot/Local_importance_{i}.png")
        fig.show()


def fit_stacking():
    X, y, points = generate_sample(
        count=5000, f=f_square_2, coef_func=[coef_auto_gau_strong, coef_auto_gau_weak], random_seed=1,
        plot=True
    )
    X_plus = np.concatenate([X, points], axis=1)
    distance_measure = "euclidean"
    kernel_type = "bisquare"

    model = StackingWeightModel(
        # ExtraTreesRegressor(n_estimators=10, max_depth=X.shape[1]),
        DecisionTreeRegressor(splitter="random", max_depth=X.shape[1]),
        distance_measure,
        kernel_type,
        neighbour_count=0.02,
        neighbour_leave_out_rate=0.2,
    )
    t1 = time.time()
    model.fit(X_plus, y, [points])
    t2 = time.time()
    print("Stacking:", model.llocv_score_, model.llocv_stacking_)
    print(t2 - t1)


if __name__ == "__main__":
    draw_graph()
    # fit_stacking()
