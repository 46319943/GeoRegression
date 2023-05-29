"""
Generate simulated data for testing purposes.
"""

import random
import time
import numpy as np
import matplotlib.pyplot as plt


def radial_coefficient(origin):
    """
    Generate a radial coefficient for a given origin.
    """

    def coefficient(point):
        return np.linalg.norm(point - origin, axis=-1)

    return coefficient


def direction_coefficient(direction):
    """
    Generate a direction coefficient for a given direction.
    """

    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)

    def coefficient(point):
        return np.dot(point, direction)

    return coefficient


def square_function(coefficient):
    return polynomial_function(coefficient, 2)


def interaction_function(coefficient):
    def function(x1, x2, point):
        return coefficient(point) * x1 * x2

    return function


def linear_function(coefficient):
    def function(x, point):
        return coefficient(point) * x

    return function


def sigmoid_function(coefficient):
    def function(x, point):
        return coefficient(point) * np.tanh(x)

    return function


def relu_function(coefficient):
    def function(x, point):
        return coefficient(point) * np.maximum(0, x)

    return function


def polynomial_function(coefficient, degree):
    def function(x, point):
        return coefficient(point) * x ** degree

    return function


def exponential_function(coefficient):
    def function(x, point):
        return coefficient(point) * np.exp(x)

    return function


def sample_points(n, dim, bounds):
    """
    Sample n points in dim dimensions from a uniform distribution with given bounds.
    The bound of each dimension can be sampled from continuous range or discrete classes.
    """

    points = np.zeros((n, dim))
    for i in range(dim):
        if isinstance(bounds[i], tuple):
            points[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], n)
        elif isinstance(bounds[i], list):
            points[:, i] = np.random.choice(bounds[i], n)

    return points


def sample_x(n, type='uniform'):
    """
    Sample n x values from a specified distribution.
    """
    if type == 'uniform':
        return np.random.uniform(-10, 10, n)
    elif type == 'normal':
        return np.random.normal(0, 1, n)
    elif type == 'exponential':
        return np.random.exponential(1, n)


def generate_sample(random_seed=None):
    np.random.seed(random_seed)

    coef1 = radial_coefficient(np.array([0, 0]))
    coef2 = direction_coefficient(np.array([1, 1]))

    points = sample_points(100, 2, [(-10, 10), (-10, 10)])

    x1 = sample_x(100)
    x2 = sample_x(100)

    y = polynomial_function(coef1, 2)(x1, points) + 3 + relu_function(coef2)(x2, points) * x1

    X = np.stack((x1, x2), axis=-1)
    coefficients = np.stack((coef1(points), coef2(points)), axis=-1)

    return X, y, points, coefficients


def show_sample(X, y, points, coefficients):
    """
    Show X, y, points, and coefficients in multiple subplots.
    Assume dimension of points is 2, which is a plane.
    """
    # Calculate the number of subplots needed.
    dim_x = X.shape[1]
    dim_coef = coefficients.shape[1]

    # Plot X. Add colorbar for each dimension.
    plt.figure()
    for i in range(dim_x):
        plt.subplot(dim_x, 1, i + 1)
        plt.scatter(points[:, 0], points[:, 1], c=X[:, i])
        plt.colorbar()
    plt.show(block=False)

    # Plot y using scatter and boxplot
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.scatter(points[:, 0], points[:, 1], c=y)

    plt.subplot(2, 1, 2)
    plt.boxplot(y)

    plt.colorbar()
    plt.show(block=False)

    # Plot coefficients
    plt.figure()
    for i in range(dim_coef):
        plt.subplot(dim_coef, 1, i + 1)
        plt.scatter(points[:, 0], points[:, 1], c=coefficients[:, i])
        plt.colorbar()
    plt.show(block=True)


def main():
    X, y, points, coefficients = generate_sample()

    show_sample(X, y, points, coefficients)


if __name__ == '__main__':
    main()
