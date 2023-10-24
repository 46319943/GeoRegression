"""
Generate simulated data for testing purposes.
A Bayesian Implementation of the Multiscale Geographically Weighted Regression Model with INLA
https://doi.org/10.1080/24694452.2023.2187756
"""
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

from georegression.test.data.simulation_utils import gaussian_coefficient, interaction_function, radial_coefficient, directional_coefficient, sample_x_across_location, sine_coefficient, coefficient_wrapper, polynomial_function, sigmoid_function, \
    sample_points, sample_x


def f_square(X, C, points):
    return (
            polynomial_function(C[0], 2)(X[:, 0], points) +
            C[0](points) * 10 +
            0
    )

def f_square_2(X, C, points):
    return (
            polynomial_function(C[0], 2)(X[:, 0], points) +
            polynomial_function(C[1], 2)(X[:, 1], points) +
            0
    )

def f_sigmoid(X, C, points):
    return (
            sigmoid_function(C[0])(X[:, 0], points) +
            0
    )

def  f_interact(X, C, points):
    return (
            interaction_function(C[0])(X[:, 0], X[:, 1], points) +
            0
    )

f = f_interact


def coef_f():
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


def coef_f2():
    coef_radial = radial_coefficient(np.array([0, 0]), 1 / np.sqrt(200))
    coef_dir = directional_coefficient(np.array([1, 1]))

    coef_gau_1 = gaussian_coefficient(np.array([-5, 5]), [[3, 4],[4, 8]], amplitude=-1)
    coef_gau_2 = gaussian_coefficient(np.array([-2, -5]), 5, amplitude=2)
    coef_gau_3 = gaussian_coefficient(np.array([8, 3]), 10, amplitude=-1.5)
    coef_gau_4 = gaussian_coefficient(np.array([2, 8]), [[3, 0], [0, 15]], amplitude=0.8)
    coef_gau_5 = gaussian_coefficient(np.array([5, -10]), 1, amplitude=1)
    coef_gau_6 = gaussian_coefficient(np.array([-10, -10]), 15, amplitude=1.5)
    coef_gau_6 = gaussian_coefficient(np.array([-11, 0]), 5, amplitude=2)
    coef_gau_6 = gaussian_coefficient(np.array([-11, 0]), 5, amplitude=2)
    coef_gau = coefficient_wrapper(np.sum, coef_gau_1, coef_gau_2, coef_gau_3, coef_gau_4, coef_gau_5, coef_gau_6)

    coef_sum = coefficient_wrapper(np.sum, coef_radial, coef_dir, coef_gau)

    return coef_sum


def coef_f3():
    coef_radial = radial_coefficient(np.array([0, 0]), 1 / np.sqrt(200) * 10)
    coef_dir = directional_coefficient(np.array([1, 1]))

    coef_sum = coefficient_wrapper(np.sum, coef_radial, coef_dir)

    return coef_sum

def coef_f4():
    # Random seed 1
    np.random.seed(1)

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
        cov = np.random.uniform(- np.sqrt(sigma1 * sigma2), np.sqrt(sigma1 * sigma2))
        sigma = np.array([[sigma1, cov], [cov, sigma2]])

        coef_gau = gaussian_coefficient(center, sigma, amplitude=amplitude)
        gau_coef_list.append(coef_gau)
        
    coef_gau = coefficient_wrapper(np.sum, *gau_coef_list)
    coef_sum = coefficient_wrapper(np.sum, coef_radial, coef_dir, coef_gau)

    return coef_sum


def coef_f5():
    # Random seed 1
    # np.random.seed(1)

    coef_radial = radial_coefficient(np.array([0, 0]), 1 / np.sqrt(200))
    coef_dir = directional_coefficient(np.array([1, 1]))

    gau_coef_list = []
    for i in range(1000):
        # Randomly generate the parameters for gaussian coefficient
        center = np.random.uniform(-10, 10, 2)
        amplitude = np.random.uniform(1, 2)
        sign = np.random.choice([-1, 1])
        amplitude *= sign
        sigma1 = np.random.uniform(0.2, 1)
        sigma2 = np.random.uniform(0.2, 1)
        cov = np.random.uniform(- np.sqrt(sigma1 * sigma2), np.sqrt(sigma1 * sigma2))
        sigma = np.array([[sigma1, cov], [cov, sigma2]])

        coef_gau = gaussian_coefficient(center, sigma, amplitude=amplitude)
        gau_coef_list.append(coef_gau)
        
    coef_gau = coefficient_wrapper(np.sum, *gau_coef_list)
    coef_sum = coefficient_wrapper(np.sum, coef_radial, coef_dir, coef_gau)

    return coef_sum


coef_func = coef_f2


def generate_sample(random_seed=None, count=100, f=f, function_coef_num=1, coef_func=coef_func):
    np.random.seed(random_seed)

    points = sample_points(count)

    # x1 = sample_x(count)
    x1 = sample_x_across_location(count, points, bound_coef=coef_func(), bounds=(-1, 1))
    # x2 = sample_x(count)
    # x2 = sample_x(count, bounds=(0, 1))
    x2 = sample_x_across_location(count, points, bound_coef=coefficient_wrapper(partial(np.multiply, 3) ,coef_func()), bounds=(-2, 2))

    coefficients = [coef_func() for _ in range(function_coef_num)]

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
        plt.subplot(dim_x, 1, i + 1)
        plt.scatter(points[:, 0], points[:, 1], c=X[:, i])
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"The {i}-th feature of X")
    plt.suptitle("The value of X across the plane")

    # Plot y using scatter and boxplot
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(points[:, 0], points[:, 1], c=y)
    plt.title("The value of y across the plane")

    plt.subplot(1, 2, 2)
    plt.boxplot(y)
    plt.title("The distribution of y")

    plt.colorbar()

    # Plot coefficients
    plt.figure()
    for i in range(dim_coef):
        plt.subplot(dim_coef, 1, i + 1)
        plt.scatter(points[:, 0], points[:, 1], c=coefficients[i](points), cmap='Spectral')
        plt.colorbar()
        plt.title(f"The {i}-th coefficient across the plane")
    plt.suptitle("The value of coefficients across the plane")


def show_function_at_point(function, coef, point, X_bounds=(-10, 10), ax=None):
    """
    Show the function value at a given point.
    """

    # Generate the x values
    x1 = np.linspace(X_bounds[0], X_bounds[1], 1000)
    x2 = np.linspace(X_bounds[0], X_bounds[1], 1000)

    X = np.stack((x1, x2), axis=-1)

    # Calculate the function value
    y = function(X, coef, point)

    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"The function value at point {point}")
        
    # Plot the function
    ax.plot(x1, y, label="Function value")
    
    return ax


def main():
    X, y, points, f, coefficients = generate_sample(count=5000, random_seed=1)

    show_sample(X, y, points, coefficients)
    show_function_at_point(f, coefficients, points[0])

    plt.show(block=True)


if __name__ == '__main__':
    main()
