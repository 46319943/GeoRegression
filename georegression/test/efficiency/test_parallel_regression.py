from time import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge


def test_parallel_regression():
    # Generate stacking data
    neighbour_count = 100
    X = np.random.random((neighbour_count, neighbour_count))
    y = np.random.random(neighbour_count)
    estimator_count = 20000

    def fit_stacking():
        return Ridge().fit(X, y)

    t_start = time()
    job_list = [delayed(fit_stacking)() for i in range(estimator_count)]
    Parallel(n_jobs=-1)(job_list)
    t_end = time()

    print(t_end - t_start)
    # neighbour_count = 500 estimator_count = 100 1.5465185642242432
    # neighbour_count = 1000 estimator_count = 1000 7.498879909515381
    # neighbour_count = 20000 estimator_count = 500 21.9149112701416
    # neighbour_count = 100 estimator_count = 20000 1.5993006229400635


    t_start = time()
    for i in range(estimator_count):
        e = Ridge()
        e.fit(X, y)
    t_end = time()

    print(t_end - t_start)
    # neighbour_count = 1000 estimator_count = 1000 19.928532361984253
    # neighbour_count = 20000 estimator_count = 500 97.64331030845642
    # neighbour_count = 100 estimator_count = 20000 10.334396123886108




if __name__ == "__main__":
    test_parallel_regression()
