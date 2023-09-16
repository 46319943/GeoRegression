import numpy as np
from numba import njit, float64, boolean


@njit()
def mean(x, axis, weight):
    weight = weight.reshape((-1, 1))
    x = x * weight
    return np.sum(x, axis) / np.sum(weight)

@njit()
def ridge_cholesky(X, y, alpha, weight):
    y = y.reshape((-1, 1))

    # Center the data to make the intercept term zero
    # (n,)
    X_offset = mean(X, 0, weight)
    # (1,)
    y_offset = mean(y, 0, weight)

    # (m, n)
    X_center = X - X_offset
    # (m,)
    y_center = y - y_offset

    # sample_weight via a simple rescaling
    weight_sqrt = np.sqrt(weight)
    for index, weight in enumerate(weight_sqrt):
        X_center[index] *= weight
        y_center[index] *= weight

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

@njit()
def r2_score(y_true, y_pred):
    # https://github.com/jcatankard/NumbaML/blob/main/numbaml/scoring.py
    """https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score"""
    y_mean = np.mean(y_true)
    total_sum_squares = np.sum((y_true - y_mean) ** 2)
    residual_sum_squares = np.sum((y_true - y_pred) ** 2)

    if total_sum_squares == 0:
        return 1

    return 1 - (residual_sum_squares / total_sum_squares)


if __name__ == '__main__':
    X = np.random.randn(1000, 100)
    y = np.random.randn(1000)
    alpha = 10
    weight = np.random.random((1000, ))

    coef, intercept = ridge_cholesky(X, y, alpha, weight)

    print(coef, intercept)
    print(coef.shape, intercept.shape)

    print()
