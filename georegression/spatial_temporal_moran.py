import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


def calculate_spatial_temporal_global_moran(y, spatial_temporal_weight):
    N = y.shape[0]
    y_mean = np.mean(y)
    z = y - y_mean
    a = np.matmul(spatial_temporal_weight, z)
    numerator = N * np.matmul(z, a)
    W = np.sum(spatial_temporal_weight)
    denominator = W * np.sum(z ** 2)
    I = numerator / denominator

    expectation = -1 / (N - 1)

    S1 = (1 / 2) * np.sum((spatial_temporal_weight + spatial_temporal_weight.T) ** 2)
    S2 = np.sum(np.sum(spatial_temporal_weight + spatial_temporal_weight.T, axis=1) ** 2)
    S3 = N ** (-1) * np.sum(z ** 4) / (N ** (-1) * np.sum(z ** 2)) ** 2
    S4 = (N ** 2 - 3 * N + 3) * S1 - N * S2 + 3 * W ** 2
    S5 = (N ** 2 - N) * S1 - 2 * N * S2 + 6 * W ** 2

    variance = (N * S4 - S3 * S5) / ((N - 1) * (N - 2) * (N - 3) * W ** 2) - expectation ** 2
    p = norm.cdf((I - expectation) / np.sqrt(variance))

    return I, expectation, variance, p


def calculate_spatial_temporal_local_moran(y, spatial_temporal_weight):
    N = y.shape[0]
    u = (y - np.mean(y)) / np.var(y)
    a = np.matmul(spatial_temporal_weight, u)
    return u * a


def plot_moran_diagram(y, weight_matrix):
    y_center = y - np.mean(y)
    y_neighbour = np.matmul(weight_matrix, y_center)
    plt.scatter(y_center, y_neighbour, s=10, edgecolors="k", alpha=0.5)
    plt.axhline(y=0, color='k', linewidth=1)  # added because i want the origin
    plt.axvline(x=0, color='k', linewidth=1)

    y_center_min = np.min(y_center)
    y_center_max = np.max(y_center)
    y_center_range = np.array([y_center_min, y_center_max]).reshape(-1, 1)

    estimator = LinearRegression(fit_intercept=False)
    estimator.fit(y_center.reshape(-1, 1), y_neighbour)
    # coef = np.polyfit(y_center, y_neighbour, 1)
    # poly1d_fn = np.poly1d(coef)
    # poly1d_fn is now a function which takes in x and returns an estimate for y
    plt.plot(y_center_range, estimator.predict(y_center_range),
             '--k')  # '--k'=black dashed line, 'yo' = yellow circle marker

    return
