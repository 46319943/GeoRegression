from scipy.sparse import csr_array
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.metrics import r2_score

from georegression.stacking_model import StackingWeightModel
from georegression.test.data import load_HP
from georegression.weight_matrix import weight_matrix_from_points
from georegression.weight_model import WeightModel

from time import time as t

X, y_true, xy_vector, time = load_HP()


def test_performance():
    weight_matrix = weight_matrix_from_points(
        [xy_vector, time],
        [xy_vector, time],
        "euclidean",
        "bisquare",
        None,
        None,
        0.1,
        None,
    )
    weight_matrix = csr_array(weight_matrix)

    estimator = StackingWeightModel(
        ExtraTreesRegressor(max_depth=1, n_estimators=100),
        neighbour_leave_out_rate=0.1,
        use_numba=True,
    )

    estimator.fit(X, y_true, [xy_vector, time], weight_matrix=weight_matrix)
    print(estimator.llocv_score_)
    print(estimator.llocv_stacking_)
    # Time taken to predict in single thread: 19.013251066207886
    # Time taken to predict in parallel: 43.77483797073364
    # 0.7931079729983194
    # 0.823086329159848


if __name__ == "__main__":
    test_performance()
