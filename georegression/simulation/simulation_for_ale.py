
from functools import partial

import numpy as np
from georegression.simulation.simulation import coef_manual_gau
from georegression.simulation.simulation_utils import coefficient_wrapper, interaction_function, sample_points, sample_x


def f_interact(X, C, points):
    return (
            interaction_function(C[0])(X[:, 0], X[:, 1], points) +
            0
    )

f = f_interact
coef_func = coef_manual_gau
x2_coef = coefficient_wrapper(partial(np.multiply, 3) ,coef_func())

def generate_sample(random_seed=None, count=100, f=f, coef_func=coef_func):
    np.random.seed(random_seed)

    points = sample_points(count)

    # x1 = sample_x(count)
    x1 = sample_x(count, mean=coef_func(), bounds=(-1, 1), points=points)

    # x2 = sample_x(count)
    # x2 = sample_x(count, bounds=(0, 1))
    x2_coef = coefficient_wrapper(partial(np.multiply, 3) ,coef_func())
    x2 = sample_x(count, mean=x2_coef, bounds=(-2, 2), points=points)
    
    if isinstance(coef_func, list):
        coefficients = [func() for func in coef_func]
    else:
        coefficients = [coef_func()]

    X = np.stack((x1, x2), axis=-1)
    y = f(X, coefficients, points)

    return X, y, points, f, coefficients
