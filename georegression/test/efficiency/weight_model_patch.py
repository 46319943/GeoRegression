from time import time as t

from scipy.sparse import csr_array
from sklearn.tree import ExtraTreeRegressor

from georegression.test.data import load_HP
from georegression.weight_matrix import weight_matrix_from_points
from georegression.weight_model import WeightModel

X, y_true, xy_vector, time = load_HP()


def test_performance_n_jobs():
    weight_matrix = weight_matrix_from_points(
        [xy_vector, time],
        [xy_vector, time],
        "euclidean",
        "bisquare",
        neighbour_count=0.1,
    )

    t1 = t()
    estimator = WeightModel(
        ExtraTreeRegressor(max_depth=1, splitter="random"),
        n_jobs=-1
    )
    estimator.fit(X, y_true, [xy_vector, time], weight_matrix=weight_matrix)
    t2 = t()
    print(t2 - t1)
    print(estimator.llocv_score_)

    weight_matrix = csr_array(weight_matrix)

    t1 = t()
    estimator = WeightModel(
        ExtraTreeRegressor(max_depth=1, splitter="random"),
        n_jobs=-1
    )
    estimator.fit(X, y_true, [xy_vector, time], weight_matrix=weight_matrix)
    t2 = t()
    print(t2 - t1)
    print(estimator.llocv_score_)

    # 5.687727212905884
    # 0.7720929346825464
    # 1.841477394104004
    # 0.7732306074996614

def test_performance_n_patches():
    weight_matrix = weight_matrix_from_points(
        [xy_vector, time],
        [xy_vector, time],
        "euclidean",
        "bisquare",
        neighbour_count=0.1,
    )
    
    t1 = t()
    estimator = WeightModel(ExtraTreeRegressor(max_depth=1, splitter="random"), n_patches=6)
    estimator.fit(X, y_true, [xy_vector, time], weight_matrix=weight_matrix)
    t2 = t()
    print(t2 - t1)
    print(estimator.llocv_score_)
    
    weight_matrix = csr_array(weight_matrix)
    
    t1 = t()
    estimator = WeightModel(ExtraTreeRegressor(max_depth=1, splitter="random"), n_patches=6)
    estimator.fit(X, y_true, [xy_vector, time], weight_matrix=weight_matrix)
    t2 = t()
    print(t2 - t1)
    print(estimator.llocv_score_)

    # 2.56437611579895
    # 0.7760204788183097
    # 0.9893820285797119
    # 0.7697055618854263

if __name__ == "__main__":
    test_performance_n_jobs()
    test_performance_n_patches()
