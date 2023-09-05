import numpy as np
from numba import jit, njit, prange
from sklearn.linear_model import Ridge
from time import time


def loop_python(iteration_count=1000):
    X = np.random.random((100, 100))
    y = np.random.random((100, 1))

    for i in range(iteration_count):
        estimator = Ridge(1)
        estimator.fit(X, y)


@jit(forceobj=True, looplift=True)
def loop_jitting(iteration_count=1000):
    X = np.random.random((100, 100))
    y = np.random.random((100, 1))

    for i in range(iteration_count):
        estimator = Ridge(1)
        estimator.fit(X, y)


@njit()
def loop_numba(iteration_count=1000):
    X = np.random.random((100, 100))
    y = np.random.random((100, 1))

    for i in range(iteration_count):
        ridge_fit(X, y)


@njit(parallel=True)
def loop_paralle(iteration_count=1000):
    X = np.random.random((100, 100))
    y = np.random.random((100, 1))

    # Stuck when X, y is passed to ridge_fit. Everything is fine if X, y are generated inside ridge_fit.

    for i in prange(iteration_count):
        ridge_fit(X, y)


# @njit(parallel=True)
def loop_paralle_lstsq(iteration_count=1000):
    X = np.random.random((100, 100))
    y = np.random.random((100, 1))

    for i in prange(iteration_count):
        alpha = 1.0

        # Center the data to make the intercept term zero
        X_offset = mean(X, axis=0)
        y_offset = mean(y, axis=0)
        X_center = X - X_offset
        y_center = y - y_offset

        dimension = X_center.shape[1]
        A = np.identity(dimension)

        X_aug = np.vstack((X_center, np.sqrt(alpha) * A))
        y_aug = np.vstack((y_center, np.zeros((A.shape[0], 1), dtype=y.dtype)))

        coef = np.linalg.lstsq(X_aug, y_aug)[0]
        intercept = y_offset - np.dot(X_offset, coef)


@njit(parallel=True)
def loop_parallel_chol(iteration_count=1000):
    X = np.random.random((100, 100))
    y = np.random.random((100, 1))

    for i in prange(iteration_count):
        ridge_cholesky(X, y)

        # alpha = 1.0
        #
        # # Center the data to make the intercept term zero
        # X_offset = mean(X, axis=0)
        # y_offset = mean(y, axis=0)
        # X_center = X - X_offset
        # y_center = y - y_offset
        #
        # A = np.dot(X_center.T, X_center)
        # Xy = np.dot(X_center.T, y_center)
        #
        # A = A + alpha * np.eye(X.shape[1])
        #
        # coef = np.linalg.solve(A, Xy)
        # intercept = y_offset - np.dot(X_offset, coef)

@njit(parallel=True)
def loop_parallel_inner(iteration_count=1000):
    X = np.random.random((100, 100))
    y = np.random.random((100, 1))

    # coef_list = []
    # intercept_list = []

    for i in prange(iteration_count):
        alpha = 1.0
        
        # Center the data to make the intercept term zero
        # X_offset = mean(X, axis=0)
        # y_offset = mean(y, axis=0)
        # X_center = X - X_offset
        # y_center = y - y_offset

        X_center = X
        y_center = y
        
        dimension = X_center.shape[1]
        A = np.identity(dimension)
        A_biased = alpha * A

        temp = X_center.T.dot(X_center) + A_biased

        np.linalg.inv(
            temp
        )

        # coef = np.linalg.inv(
        #     X_center.T.dot(X_center) + A_biased
        # ).dot(X_center.T).dot(y_center)
        # intercept = y_offset - np.dot(X_offset, coef)
        
        # coef_list.append(coef)
        # intercept_list.append(intercept)


@njit()
def mean(x, axis):
    return np.sum(x, axis) / x.shape[axis]


@njit()
def ridge_fit(X, y):
    alpha = 1.0

    # Center the data to make the intercept term zero
    X_offset = mean(X, axis=0)
    y_offset = mean(y, axis=0)
    X_center = X - X_offset
    y_center = y - y_offset

    dimension = X_center.shape[1]
    A = np.identity(dimension)
    A_biased = alpha * A

    coef = np.linalg.inv(X_center.T.dot(X_center) + A_biased).dot(X_center.T).dot(y_center)
    intercept = y_offset - np.dot(X_offset, coef)

    return coef, intercept


@njit()
def rigde_lstsq(X, y):
    alpha = 1.0

    # Center the data to make the intercept term zero
    X_offset = mean(X, axis=0)
    y_offset = mean(y, axis=0)
    X_center = X - X_offset
    y_center = y - y_offset

    dimension = X_center.shape[1]
    A = np.identity(dimension)

    X_aug = np.vstack((X_center, np.sqrt(alpha) * A))
    y_aug = np.vstack((y_center, np.zeros((A.shape[0], 1), dtype=y.dtype)))

    coef = np.linalg.lstsq(X_aug, y_aug)[0]
    intercept = y_offset - np.dot(X_offset, coef)

    return coef, intercept

@njit()
def ridge_cholesky(X, y):
    alpha = 1.0

    # Center the data to make the intercept term zero
    X_offset = mean(X, axis=0)
    y_offset = mean(y, axis=0)
    X_center = X - X_offset
    y_center = y - y_offset

    A = np.dot(X_center.T, X_center)
    Xy = np.dot(X_center.T, y_center)

    A = A + alpha * np.eye(X.shape[1])

    coef = np.linalg.solve(A, Xy)
    intercept = y_offset - np.dot(X_offset, coef)

    return coef, intercept


def test_ridge_work():
    X = np.random.random((10000, 1000))
    y = np.random.random((10000, 1))

    ridge_fit(X, y)
    t1 = time()
    coef, intercept = ridge_fit(X, y)
    t2 = time()
    print(coef, intercept)

    t3 = time()
    estimator = Ridge(1.0).fit(X, y)
    t4 = time()
    print(estimator.coef_, estimator.intercept_)

    rigde_lstsq(X, y)
    t5 = time()
    coef, intercept = rigde_lstsq(X, y)
    t6 = time()
    print(coef, intercept)

    coef, intercept = ridge_cholesky(X, y)
    t7 = time()
    coef, intercept = ridge_cholesky(X, y)
    t8 = time()
    print(coef, intercept)


    print(t2 - t1)
    print(t4 - t3)
    print(t6 - t5)
    print(t8 - t7)


def test_loop():

    t1 = time()
    loop_python(100000)
    t2 = time()
    print(t2 - t1)

    # loop_jitting()
    t1 = time()
    # loop_jitting()
    t2 = time()
    print(t2 - t1)

    # loop_numba()
    t1 = time()
    # loop_numba()
    t2 = time()
    print(t2 - t1)

    # loop_paralle()
    t1 = time()
    # loop_paralle()
    t2 = time()
    print(t2 - t1)

    # loop_parallel_inner()
    t1 = time()
    # loop_parallel_inner()
    t2 = time()
    print(t2 - t1)

    # loop_paralle_lstsq(1)
    t1 = time()
    # loop_paralle_lstsq(10000)
    t2 = time()
    print(t2 - t1)

    loop_parallel_chol(1)
    t1 = time()
    loop_parallel_chol(100000)
    t2 = time()
    print(t2 - t1)

if __name__ == "__main__":
    test_ridge_work()

    # 41 for intel extension
    # 39 for original sklearn
    test_loop()
