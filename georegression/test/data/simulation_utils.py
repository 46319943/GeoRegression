import numpy as np
from scipy.stats import multivariate_normal

def gaussian_coefficient(mean, cov, amplitude=1):
    """
    Generate a gaussian coefficient for a given mean and variance.
    """

    if isinstance(cov, list):
        cov = np.array(cov)

    normalize_factor = multivariate_normal.pdf(mean, mean, cov)

    def coefficient(point):
        return multivariate_normal.pdf(point, mean, cov) * amplitude / normalize_factor

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


def sample_x_across_location(n, points, type='uniform', bound_coef=None, bounds=(-10, 10), mean=0, variance=1, scale=1):
    """
    Sample n x values from a specified distribution across a given location.
    """

    if callable(bound_coef):
            x = np.zeros((n, ))
            for i in range(points.shape[0]):
                base_bound = bound_coef(points[i])
                x[i] = np.random.uniform(base_bound + bounds[0], base_bound + bounds[1], 1)

            return x
        

    # if isinstance(bounds[0], function):
    #     low_bound_coef = bounds[0]
    # if isinstance(bounds[1], function):
    #     high_bound_coef = bounds[1]

    # x = np.zeros((n, ))

    # for i in range(points.shape[1]):
    #     x[i] = np.random.uniform(low_bound_coef(points[i]), high_bound_coef(points[i]), 1)

    # return x