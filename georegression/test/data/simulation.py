"""
Generate simulated data for testing purposes.
A Bayesian Implementation of the Multiscale Geographically Weighted Regression Model with INLA
https://doi.org/10.1080/24694452.2023.2187756
"""
import matplotlib.pyplot as plt
import numpy as np

from georegression.test.data.simulation_utils import gaussian_coefficient, radial_coefficient, directional_coefficient, sine_coefficient, coefficient_wrapper, polynomial_function, sigmoid_function, \
    sample_points, sample_x


def f_square(X, C, points):
    return (
            polynomial_function(C[0], 2)(X[:, 0], points) +
            0
    )

def f_square_2(X, C, points):
    return (
            polynomial_function(C[0], 2)(X[:, 0], points) +
            polynomial_function(C[0], 2)(X[:, 1], points) +
            0
    )

def f_sigmoid(X, C, points):
    return (
            sigmoid_function(C[0])(X[:, 0], points) +
            0
    )

f = f_square


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
    coef_sin_1 = sine_coefficient(0.8, np.array([-1, 1]), 0)
    coef_sin_2 = sine_coefficient(0.6, np.array([1, 1]), 0)
    coef_sin = coefficient_wrapper(np.sum, coef_sin_1, coef_sin_2)
    coef_gau_1 = gaussian_coefficient(np.array([-5, 5]), 3, amplitude=-1)
    coef_gau_2 = gaussian_coefficient(np.array([-2, -5]), 5, amplitude=2)
    coef_gau = coefficient_wrapper(np.sum, coef_gau_1, coef_gau_2)

    coef_sum = coefficient_wrapper(np.sum, coef_radial, coef_dir, coef_sin, coef_gau)

    return coef_sum

def coef_f3():
    coef_radial = radial_coefficient(np.array([5, 5]), 1 / np.sqrt(200))
    coef_dir = directional_coefficient(np.array([1, 1]))
    coef_sin_1 = sine_coefficient(0.8, np.array([-1, 1]), 0)
    coef_sin_2 = sine_coefficient(0.6, np.array([1, 1]), 0)
    coef_sin = coefficient_wrapper(np.sum, coef_sin_1, coef_sin_2)
    coef_gau_1 = gaussian_coefficient(np.array([-5, 5]), 3, amplitude=-1)
    coef_gau_2 = gaussian_coefficient(np.array([-2, -5]), 5, amplitude=2)
    coef_gau = coefficient_wrapper(np.sum, coef_gau_1, coef_gau_2)

    coef_sum = coefficient_wrapper(np.sum, coef_radial, coef_dir, coef_sin, coef_gau)

    return coef_sum

coef = coef_f()

def generate_sample(random_seed=None, count=100, f=f, coef=coef):
    np.random.seed(random_seed)

    points = sample_points(count)

    x1 = sample_x(count)
    x2 = sample_x(count)
    coefficients = [coef]

    X = np.stack((x1, x2), axis=-1)
    y = f(X, coefficients, points)

    return X, y, points, coefficients


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
    plt.show(block=False)
    plt.suptitle("The value of X across the plane")

    # Plot y using scatter and boxplot
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.scatter(points[:, 0], points[:, 1], c=y)
    plt.title("The value of y across the plane")

    plt.subplot(2, 1, 2)
    plt.boxplot(y)
    plt.title("The distribution of y")

    plt.colorbar()
    plt.show(block=False)

    # Plot coefficients
    plt.figure()
    for i in range(dim_coef):
        plt.subplot(dim_coef, 1, i + 1)
        plt.scatter(points[:, 0], points[:, 1], c=coefficients[i](points))
        plt.colorbar()
        plt.title(f"The {i}-th coefficient across the plane")
    plt.show(block=False)
    plt.suptitle("The value of coefficients across the plane")


def show_function_at_point(function, coef, point, X_bounds=(-10, 10)):
    """
    Show the function value at a given point.
    """

    # Generate the x values
    x1 = np.linspace(X_bounds[0], X_bounds[1], 1000)
    x2 = np.linspace(X_bounds[0], X_bounds[1], 1000)

    X = np.stack((x1, x2), axis=-1)

    # Calculate the function value
    y = function(X, coef, point)

    # Plot the function
    plt.figure()
    plt.plot(x1, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"The function value at point {point}")
    plt.show(block=False)


def main():
    X, y, points, coefficients = generate_sample(count=5000, random_seed=1)

    show_sample(X, y, points, coefficients)
    show_function_at_point(f, coefficients, points[0])

    plt.figure()
    plt.show(block=True)


if __name__ == '__main__':
    main()
