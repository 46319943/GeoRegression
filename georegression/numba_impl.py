import numpy as np
from numba import njit


@njit()
def mean(x, axis):
    return np.sum(x, axis) / x.shape[axis]

@njit()
def ridge_cholesky(X, y, alpha):
    # Center the data to make the intercept term zero

    # (n,)
    X_offset = np.sum(X, 0) / X.shape[0]
    # (1,)
    y_offset = np.sum(y, 0) / y.shape[0]

    # (m, n)
    X_center = X - X_offset
    # (m,)
    y_center = y - y_offset

    # (n, n)
    A = np.dot(X_center.T, X_center)
    # (n, 1)
    Xy = np.dot(X_center.T, y_center)

    A = A + alpha * np.eye(X.shape[1])

    # (n,)
    coef = np.linalg.solve(A, Xy)
    # (1,)
    intercept = y_offset - np.dot(X_offset, coef)

    return coef, intercept


if __name__ == '__main__':
    X = np.random.randn(1000, 100)
    y = np.random.randn(1000)
    alpha = 10

    coef, intercept = ridge_cholesky(X, y, alpha)

    print(coef, intercept)
    print(coef.shape, intercept.shape)
