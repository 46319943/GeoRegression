import os
from functools import partial

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from georegression.local_ale import weighted_ale
from georegression.simulation.simulation import show_sample
from georegression.simulation.simulation_utils import *
from georegression.visualize.ale import plot_ale
from georegression.weight_model import WeightModel


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


def f_interact(X, C, points):
    return interaction_function(C[0])(X[:, 0], X[:, 1], points) + 0


def generate_sample(random_seed=1):
    np.random.seed(random_seed)

    count = 5000
    points = sample_points(count, bounds=[[-10, 10], [-10, 10]])

    coef_x1 = coef_auto_gau_weak()
    coef_x2 = coefficient_wrapper(partial(np.multiply, 3), coef_x1)
    x1 = sample_x(count, mean=coef_x1, bounds=(-1, 1), points=points)
    x2 = sample_x(count, mean=coef_x2, bounds=(-2, 2), points=points)

    f = f_interact
    coef_func = coef_auto_gau_strong
    if isinstance(coef_func, list):
        coefficients = [func() for func in coef_func]
    else:
        coefficients = [coef_func()]

    X = np.stack((x1, x2), axis=-1)
    y = f(X, coefficients, points)

    folder = f"Plot/ALE_{coef_func.__name__}_{f.__name__}_{count}"
    os.makedirs(folder, exist_ok=True)
    show_sample(X, y, points, coefficients, folder)

    return X, y, points, f, coefficients


def draw_graph():
    X, y, points, f, coef = generate_sample()
    X_plus = np.concatenate([X, points], axis=1)
    distance_measure = "euclidean"
    kernel_type = "bisquare"
    neighbour_count = 0.03

    # local_estimator = DecisionTreeRegressor(splitter="random", max_depth=1)
    # local_estimator = DecisionTreeRegressor(splitter="random", max_depth=2)
    # model = StackingWeightModel(
    #     local_estimator,
    #     distance_measure,
    #     kernel_type,
    #     neighbour_count=neighbour_count,
    #     neighbour_leave_out_rate=0.25,
    #     cache_data=True,
    #     cache_estimator=True,
    # )
    # model.fit(X, y, [points])
    # print('Stacking:', model.llocv_score_, model.llocv_stacking_)

    model = WeightModel(
        RandomForestRegressor(n_estimators=50),
        distance_measure,
        kernel_type,
        neighbour_count=neighbour_count,
        # cache_data=False,
        # cache_estimator=False,
        cache_data=True,
        cache_estimator=True,
    )
    model.fit(X_plus, y, [points])
    print("GRF:", model.llocv_score_)

    X = X_plus

    feature_index = 0

    for local_index in range(model.N):
        estimator = model.local_estimator_list[local_index]
        neighbour_mask = model.neighbour_matrix_[local_index]
        neighbour_weight = model.weight_matrix_[local_index][neighbour_mask]
        X_local = model.X[neighbour_mask]
        ale_result = weighted_ale(
            X_local, feature_index, estimator.predict, neighbour_weight
        )

        fval, ale = ale_result

        x_neighbour = X[model.neighbour_matrix_[local_index], feature_index]
        y_neighbour = y[model.neighbour_matrix_[local_index]]
        weight_neighbour = model.weight_matrix_[
            local_index, model.neighbour_matrix_[local_index]
        ]

        # show_function_at_point(f, coef, points[local_index], ax=ax)
        # Get the true marginal effect for function f = x1 * x2.
        x_gird = np.linspace(np.min(x_neighbour), np.max(x_neighbour), 1000)
        x1 = X[local_index, 1]
        x1 = np.tile(x1, 1000)

        # X_grid = np.stack([x_gird, x1], axis=-1)
        # y_grid = f(X_grid, coef, points[local_index])

        from georegression.simulation.simulation import x2_coef

        beta = coef[0](points[local_index])
        x2_average = x2_coef(points[local_index])
        y_grid = (
            beta
            * 0.5
            * ((x2_average + 2) ** 2 - (x2_average - 2) ** 2)
            * (1 / 4)
            * x_gird
        )

        x1_base = np.empty(500)
        x1_base[:] = np.min(x_neighbour)
        x2_base = np.random.uniform(x2_average - 2, x2_average + 2, 500)
        # base_value_real = estimator.predict(
        #     np.stack([x1_base, x2_base, points], axis=-1)
        # ).mean()
        #
        # diff = ale[0] - base_value_real
        # ale = ale - diff

        fig = plot_ale(fval, ale, x_neighbour)
        ax = fig.get_axes()[0]
        scatter = ax.scatter(x_neighbour, y_neighbour, c=weight_neighbour)
        ax.scatter(
            X[local_index, feature_index], y[local_index], c="red", label="Local point"
        )
        fig.colorbar(scatter, ax=ax, label="Weight")

        ax.plot(x_gird, y_grid, label="Function value")

        # Neighbor ALE, which only consider the neighbor points but not weight is considered
        # ale_result = weighted_ale(
        #     X_local, feature_index, estimator.predict, np.ones(X_local.shape[0])
        # )
        # fval, ale = ale_result
        # diff = ale[0] - base_value_real
        # ale = ale - diff
        # ax.plot(fval, ale, label="Neighbour ALE")

        # Add non-weighted ALE plot
        # Select the X that is in the value range of x_neighbour
        x_global_ale = X[
            (X[:, feature_index] >= np.min(x_neighbour))
            & (X[:, feature_index] <= np.max(x_neighbour))
        ]
        ale_result = weighted_ale(
            x_global_ale,
            feature_index,
            estimator.predict,
            np.ones(x_global_ale.shape[0]),
        )
        fval, ale = ale_result

        x1_base = np.empty(500)
        x1_base[:] = np.min(x_neighbour)
        x2_base = np.random.choice(X[:, 1], 500)
        # base_value_real = estimator.predict(
        #     np.stack([x1_base, x2_base, points], axis=-1)
        # ).mean()
        #
        # diff = ale[0] - base_value_real
        # ale = ale - diff

        ax.plot(fval, ale, label="Non-weighted ALE")

        # ALE True value
        # y_grid_ale = (x2_coef(points[local_index])* x_gird)
        # ax.plot(x_gird, y_grid_ale, label="True ALE")

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        handles.append(scatter)
        labels.append("Weight")
        ax.legend(handles, labels)

        plt.show(block=True)


if __name__ == "__main__":
    draw_graph()
