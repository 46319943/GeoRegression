"""
Generate simulated data for testing purposes.
A Bayesian Implementation of the Multiscale Geographically Weighted Regression Model with INLA
https://doi.org/10.1080/24694452.2023.2187756
"""
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

from georegression.simulation.simulation_utils import (
    gaussian_coefficient,
    interaction_function,
    radial_coefficient,
    directional_coefficient,
    sine_coefficient,
    coefficient_wrapper,
    polynomial_function,
    sigmoid_function,
    sample_points,
    sample_x,
)


def f_interact(X, C, points):
    return interaction_function(C[0])(X[:, 0], X[:, 1], points) + 0


def f_square_2(X, C, points):
    return (
        polynomial_function(C[0], 2)(X[:, 0], points)
        + polynomial_function(C[1], 2)(X[:, 1], points)
        + 0
    )


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

    return coef_sum


f = f_interact
coef_func = coef_manual_gau
x2_coef = coefficient_wrapper(partial(np.multiply, 3), coef_func())


def generate_sample(random_seed=None, count=100, f=f, coef_func=coef_func):
    np.random.seed(random_seed)

    points = sample_points(count, bounds=[[-10, 10], [-10, 10]])

    # x1 = sample_x(count)
    x1 = sample_x(count, mean=coef_func(), bounds=(-1, 1), points=points)

    # x2 = sample_x(count)
    # x2 = sample_x(count, bounds=(0, 1))
    x2_coef = coefficient_wrapper(partial(np.multiply, 3), coef_func())
    x2 = sample_x(count, mean=x2_coef, bounds=(-2, 2), points=points)

    if isinstance(coef_func, list):
        coefficients = [func() for func in coef_func]
    else:
        coefficients = [coef_func()]

    X = np.stack((x1, x2), axis=-1)
    y = f(X, coefficients, points)

    return X, y, points, f, coefficients


def show_sample(X, y, points, coefficients):
    """
    Show X, y, points, and coefficients in multiple subplots.
    Assume dimension of points is 2, which is a plane.
    """

    dim_points = points.shape[1]
    if dim_points != 2:
        raise ValueError("Dimension of points must be 2.")

    # Calculate the number of subplots needed.
    dim_x = X.shape[1]
    dim_coef = len(coefficients)

    # Plot X. Add colorbar for each dimension.
    plt.figure()
    for i in range(dim_x):
        plt.figure()
        plt.scatter(points[:, 0], points[:, 1], c=X[:, i], cmap="Spectral")
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"The {i+1}-th feature of X")
        plt.savefig(f"Plot/Simulation_X_{i+1}.png")

    # Plot y using scatter and boxplot
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], c=y, cmap="Spectral")
    plt.colorbar()
    plt.title("The value of y across the plane")
    plt.savefig("Plot/Simulation_y.png")

    plt.figure()
    plt.boxplot(y)
    plt.title("The distribution of y")
    plt.savefig("Plot/Simulation_y_boxplot.png")

    # Plot coefficients
    for i in range(dim_coef):
        plt.figure()
        plt.scatter(
            points[:, 0], points[:, 1], c=coefficients[i](points), cmap="Spectral"
        )
        plt.colorbar()
        plt.title(f"The {i+1}-th coefficient across the plane")
        plt.savefig(f"Plot/Simulation_Coefficients_{i}.png")


def main():
    X, y, points, f, coefficients = generate_sample(count=5000, random_seed=1)

    show_sample(X, y, points, coefficients)
    plt.show(block=True)

    # Save all figure to the local directory
    # for i in plt.get_fignums():
    # plt.figure(i)
    # plt.savefig(f"figure{i}.png")


if __name__ == "__main__":
    main()
