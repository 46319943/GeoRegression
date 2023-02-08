import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from georegression.spatial_temporal_moran import calculate_spatial_temporal_global_moran, \
    calculate_spatial_temporal_local_moran, plot_moran_diagram
from georegression.test.data import load_HP, load_TOD
from georegression.weight_model import WeightModel

# (X, y_true, xy_vector, time) = load_HP()
(X, y_true, xy_vector, time) = load_TOD()


def test_moran():
    model = WeightModel(
        LinearRegression(),
        distance_measure='euclidean',
        kernel_type='bisquare',
        neighbour_count=0.1,
    )
    model.fit(X, y_true, [xy_vector, time])
    print(model.llocv_score_)

    global_moran = calculate_spatial_temporal_global_moran(
        np.abs(model.local_residual_),
        model.weight_matrix_)
    local_moran = calculate_spatial_temporal_local_moran(
        np.abs(model.local_residual_),
        model.weight_matrix_)

    print(global_moran, local_moran)


def test_moran_diagram():
    model = WeightModel(
        LinearRegression(),
        distance_measure='euclidean',
        kernel_type='bisquare',
        neighbour_count=0.5,
    )
    model.fit(X, y_true, [xy_vector, time])
    print(model.llocv_score_)

    plot_moran_diagram(y_true, model.weight_matrix)
    plt.savefig('test_moran_diagram.png')


if __name__ == '__main__':
    test_moran()
