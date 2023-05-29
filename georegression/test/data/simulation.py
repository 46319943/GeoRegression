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
        return np.sum((point - origin) ** 2, axis=-1)

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
    def function(x, point):
        return coefficient(point) * x ** 2

    return function


def sample_points(n, dim, bounds):
    """
    Sample n points in dim dimensions from a uniform distribution with given bounds.
    """

    points = np.zeros((n, dim))

    for i in range(dim):
        points[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], n)

    return points


def sample_x(n, bounds):
    """
    Sample n x values from a uniform distribution with given bounds.
    """

    return np.random.uniform(bounds[0], bounds[1], n)


def generate_sample():
    points = sample_points(100, 2, [(-10, 10), (-10, 10)])
    x = sample_x(100, (-10, 10))
    coef1 = radial_coefficient(np.array([0, 0]))
    coef2 = direction_coefficient(np.array([1, 1]))
    y = square_function(coef1)(x, points) + square_function(coef2)(x, points)

    return x, y, points, coef1, coef2


def show_sample(x, y, points):
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()


def main():
    x, y, points, coef1, coef2 = generate_sample()

    # Color points by coefficient
    plt.scatter(points[:, 0], points[:, 1], c=coef1(points))
    plt.show()
    plt.scatter(points[:, 0], points[:, 1], c=coef2(points))
    plt.show()

    # Color points by x value
    plt.scatter(points[:, 0], points[:, 1], c=x)
    plt.show()

    # Color points by y value
    plt.scatter(points[:, 0], points[:, 1], c=y)
    plt.show()



if __name__ == '__main__':
    main()
