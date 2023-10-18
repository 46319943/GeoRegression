"""
Generate simulated data for testing purposes.
A Bayesian Implementation of the Multiscale Geographically Weighted Regression Model with INLA
https://doi.org/10.1080/24694452.2023.2187756
"""
import matplotlib.pyplot as plt
import numpy as np


def gaussian_coefficient(mean, variance, amplitude=1):
    """
    Generate a gaussian coefficient for a given mean and variance.
    """

    def coefficient(point):
        return np.exp(-np.linalg.norm(point - mean, axis=-1) ** 2 / (2 * variance)) * amplitude

    return coefficient


def radial_coefficient(origin, amplitude):
    """
    Generate a radial coefficient for a given origin.
    """

    def coefficient(point):
        return np.linalg.norm(point - origin, axis=-1) * amplitude

    return coefficient


def directional_coefficient(direction, amplitude=1):
    """
    Generate a directional coefficient for a given direction.
    """

    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)

    def coefficient(point):
        return np.dot(point, direction) / np.linalg.norm(point, axis=-1) * amplitude

    return coefficient


def sine_coefficient(frequency, direction, amplitude):
    """
    Generate a sine coefficient for a given frequency.
    """

    def coefficient(point):
        return np.sin(np.dot(point, direction) * frequency) * amplitude

    return coefficient


def coefficient_wrapper(operator, *coefficients):
    """
    Wrap two coefficients with an operator.
    """
    if len(coefficients) == 1:
        def coefficient_func(point):
            return operator(coefficients[0](point))

        return coefficient_func

    elif coefficients is not None:
        def coefficient_func(point):
            return operator(np.column_stack([c(point) for c in coefficients]), axis=-1)

        return coefficient_func

    return None


def polynomial_function(coefficient, degree):
    def function(x, point):
        return coefficient(point) * x ** degree

    return function


def square_function(coefficient):
    return polynomial_function(coefficient, 2)


def linear_function(coefficient):
    return polynomial_function(coefficient, 1)


def interaction_function(coefficient):
    def function(x1, x2, point):
        return coefficient(point) * x1 * x2

    return function


def sigmoid_function(coefficient):
    def function(x, point):
        return coefficient(point) * np.tanh(x)

    return function


def relu_function(coefficient):
    def function(x, point):
        return coefficient(point) * np.maximum(0, x)

    return function


def exponential_function(coefficient):
    def function(x, point):
        return coefficient(point) * np.exp(x)

    return function


def sample_points(n, dim=2, bounds=(-10, 10), numeraical_type="continuous"):
    """
    Sample n points in dim dimensions from a uniform distribution with given bounds.
    The bound of each dimension can be sampled from continuous range or discrete classes.
    """

    points = np.zeros((n, dim))

    if not isinstance(bounds, list):
        bounds = [bounds] * dim

    if not isinstance(numeraical_type, list):
        numeraical_type = [numeraical_type] * dim

    for i in range(dim):
        if numeraical_type[i] == "continuous":
            points[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], n)
        elif numeraical_type[i] == "discrete":
            points[:, i] = np.random.choice(bounds[i], n)

    return points


def sample_x(n, type='uniform', bounds=(-10, 10), mean=0, variance=1, scale=1):
    """
    Sample n x values from a specified distribution.
    """
    if type == 'uniform':
        return np.random.uniform(bounds[0], bounds[1], n)
    elif type == 'normal':
        return np.random.normal(mean, variance, n)
    elif type == 'exponential':
        return np.random.exponential(scale, n)


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

def generate_sample(random_seed=None, count=100, f=f):
    np.random.seed(random_seed)

    coef_radial = radial_coefficient(np.array([0, 0]), 1 / np.sqrt(200))
    coef_dir = directional_coefficient(np.array([1, 1]))
    coef_sin_1 = sine_coefficient(1, np.array([-1, 1]), 1)
    coef_sin_2 = sine_coefficient(1, np.array([1, 1]), 1)
    coef_sin = coefficient_wrapper(np.sum, coef_sin_1, coef_sin_2)
    coef_gau_1 = gaussian_coefficient(np.array([-5, 5]), 3)
    coef_gau_2 = gaussian_coefficient(np.array([-5, -5]), 3, amplitude=2)
    coef_gau = coefficient_wrapper(np.sum, coef_gau_1, coef_gau_2)

    coef_sum = coefficient_wrapper(np.sum, coef_radial, coef_dir, coef_sin, coef_gau)

    points = sample_points(count)

    x1 = sample_x(count)
    x2 = sample_x(count)
    coefficients = [coef_sum]

    X = np.stack((x1, ), axis=-1)
    y = f(X, coefficients, points)


    return X, y, points, coefficients


def show_sample(X, y, points, coefficients):
    """
    Show X, y, points, and coefficients in multiple subplots.
    Assume dimension of points is 2, which is a plane.
    """
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
    X, y, points, coefficients = generate_sample(count=1000, random_seed=1)

    show_sample(X, y, points, coefficients)
    show_function_at_point(f, coefficients, points[0])

    plt.figure()
    plt.show(block=True)


if __name__ == '__main__':
    main()
