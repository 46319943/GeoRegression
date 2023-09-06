from scipy.sparse import csr_array
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from georegression.stacking_model import StackingWeightModel
from georegression.test.data import load_HP
from georegression.weight_matrix import calculate_compound_weight_matrix
from georegression.weight_model import WeightModel

X, y_true, xy_vector, time = load_HP()


def test_compatibility():
    estimator = WeightModel(
        LinearRegression(),
        "euclidean",
        "bisquare",
        neighbour_count=0.1
    )

    weight_matrix = calculate_compound_weight_matrix(
        [xy_vector, time], [xy_vector, time], "euclidean", "bisquare", None, None, 0.1, None
    )
    sparse_matrix = csr_array(weight_matrix)
    estimator.fit(X, y_true, [xy_vector, time], weight_matrix=sparse_matrix)

    print(estimator.llocv_score_)


def test_stacking_compatibility():
    estimator = StackingWeightModel(
        ExtraTreeRegressor(max_depth=1, splitter="random"),
        "euclidean",
        "bisquare",
        neighbour_count=0.05,
        neighbour_leave_out_rate=0.05,
    )

    weight_matrix = calculate_compound_weight_matrix(
        [xy_vector, time], [xy_vector, time], "euclidean", "bisquare", None, None, 0.1, None
    )
    sparse_matrix = csr_array(weight_matrix)
    estimator.fit(X, y_true, [xy_vector, time], weight_matrix=sparse_matrix)

    print(estimator.llocv_score_)
    print(estimator.llocv_stacking_)


if __name__ == '__main__':
    # test_compatibility()
    test_stacking_compatibility()
