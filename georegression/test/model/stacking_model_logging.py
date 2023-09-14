from sklearn.tree import ExtraTreeRegressor

from georegression.stacking_model import StackingWeightModel
from georegression.test.data import load_HP

X, y_true, xy_vector, time = load_HP()


def test_logging():
    distance_measure = "euclidean"
    kernel_type = "bisquare"
    distance_ratio = None
    bandwidth = None
    neighbour_count = 0.1

    estimator = StackingWeightModel(
        ExtraTreeRegressor(splitter="random", max_depth=1),
        distance_measure,
        kernel_type,
        distance_ratio,
        bandwidth,
        neighbour_count,
        neighbour_leave_out_rate=0.1,
    )

    estimator.fit(X, y_true, [xy_vector, time])


if __name__ == "__main__":
    test_logging()
