import time

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV


def test_regression():
    neighbour_count = 500
    X = np.random.random((neighbour_count, neighbour_count))
    y = np.random.random(neighbour_count)
    estimator_count = 100

    t1 = time.time()

    for i in range(estimator_count):
        e = LinearRegression()
        e.fit(X, y)

    t2 = time.time()

    for i in range(estimator_count):
        e = Ridge()
        e.fit(X, y)

    t3 = time.time()

    for i in range(estimator_count):
        e = RidgeCV()
        e.fit(X, y)

    t4 = time.time()

    print()
    print(t2 - t1, t3 - t2, t4 - t3)
    # 2.859616994857788 0.6065783500671387 3.498260974884033
    # neighbour_count = 500 estimator_count = 100 AMD-7950X 10.886837720870972 0.49491024017333984 6.453542470932007


def test_solver():
    neighbour_count = 1000
    X = np.random.random((neighbour_count, neighbour_count))
    y = np.random.random(neighbour_count)
    estimator_count = 100

    t1 = time.time()

    for i in range(estimator_count):
        e = Ridge(10)
        e.fit(X, y)

    t2 = time.time()

    for i in range(estimator_count):
        e = Ridge(10, solver="svd")
        e.fit(X, y)

    t3 = time.time()

    for i in range(estimator_count):
        e = Ridge(10, solver="cholesky")
        e.fit(X, y)

    t4 = time.time()

    for i in range(estimator_count):
        e = Ridge(10, solver="sparse_cg")
        e.fit(X, y)

    t5 = time.time()

    for i in range(estimator_count):
        e = Ridge(10, solver="lsqr")
        e.fit(X, y)

    t6 = time.time()

    for i in range(estimator_count):
        e = Ridge(10, solver="sag")
        e.fit(X, y)

    t7 = time.time()

    print()
    print(t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6)
    # neighbour = 500 0.5866434574127197 4.114291429519653 0.6661701202392578 0.4684324264526367 0.3340754508972168 9.437297821044922
    # neighbour = 1000 2.821377992630005 21.683172464370728 2.867004156112671 0.9687221050262451 0.9672167301177979 64.40824103355408


def test_alpha():
    neighbour_count = 1000
    X = np.random.random((neighbour_count, neighbour_count))
    y = np.random.random(neighbour_count)
    estimator_count = 100

    t1 = time.time()

    for i in range(estimator_count):
        e = Ridge(0)
        e.fit(X, y)

    t2 = time.time()

    for i in range(estimator_count):
        e = Ridge(0.1)
        e.fit(X, y)

    t3 = time.time()

    for i in range(estimator_count):
        e = Ridge(10)
        e.fit(X, y)

    t4 = time.time()

    print()
    print(t2 - t1, t3 - t2, t4 - t3)
    # neighbour = 1000 57.08750033378601 1.9294869899749756 1.918003797531128


if __name__ == "__main__":
    test_regression()
    # test_solver()
    # test_alpha()
