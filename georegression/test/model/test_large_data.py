from time import time as t

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.tree import DecisionTreeRegressor

from georegression.stacking_model import StackingWeightModel
from georegression.test.data import load_HP, load_ESI
from georegression.weight_model import WeightModel

X, y_true, xy_vector, time = load_ESI()


def test_large_data():
    local_estimator = DecisionTreeRegressor(splitter="random", max_depth=2)
    distance_measure = "euclidean"
    kernel_type = "bisquare"
    distance_ratio = None
    bandwidth = None
    neighbour_count = 0.01

    model = StackingWeightModel(
        local_estimator,
        distance_measure,
        kernel_type,
        distance_ratio,
        bandwidth,
        neighbour_count,
    )

    t_start = t()

    model.fit(X, y_true, [xy_vector, time])
    print(f"{model.llocv_score_}, {model.llocv_stacking_}")

    t_end = t()
    print(f"Time: {t_end - t_start}")

if __name__ == '__main__':
    test_large_data()