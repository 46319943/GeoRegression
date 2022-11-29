from sklearn.linear_model import LinearRegression

from georegression.test.data import load_HP, load_ESI
from georegression.weight_model import WeightModel
from georegression.stacking_model import StackingWeightModel

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from time import time as t

from sklearn.ensemble import StackingRegressor

# X, y_true, xy_vector, time = load_TOD()
X, y_true, xy_vector, time = load_HP()


# X, y_true, xy_vector, time = load_ESI()


def test_stacking():
    local_estimator = DecisionTreeRegressor(splitter='random', max_depth=2)
    distance_measure = 'euclidean'
    kernel_type = 'bisquare'
    distance_ratio = None
    bandwidth = None
    neighbour_count = 0.01
    midpoint = True
    p = None

    model = StackingWeightModel(
        local_estimator,
        distance_measure,
        kernel_type,
        distance_ratio,
        bandwidth,
        neighbour_count,
        midpoint, p
    )

    model.fit(X, y_true, [xy_vector, time])
    print(f'{model.llocv_score_}, {model.llocv_stacking_}')


def test_performance():
    local_estimator = DecisionTreeRegressor(splitter='random', max_depth=2)
    distance_measure = 'euclidean'
    kernel_type = 'bisquare'
    distance_ratio = None
    bandwidth = None
    neighbour_count = 0.01
    midpoint = True
    p = None

    geographic_nearest_neighbour = 46
    temporal_nearest_neighbour = 14

    estimator = WeightModel(
        local_estimator,
        distance_measure,
        kernel_type,
        distance_ratio,
        bandwidth,
        neighbour_count,
        midpoint, p
    )

    estimator.n_jobs = -1

    t1 = t()
    estimator.use_stacking = True
    estimator.fit(X, y_true, [xy_vector, time])
    t2 = t()
    print(t2 - t1, estimator.llocv_score_, estimator.llocv_stacking_)
    # 917.4381189346313 0.7218072208798815 0.8274797786820569
    # 185.95322585105896 0.7314718925728154 0.8160008434522774
    # 471.748841047287 0.7170324025502195 0.8186682469210496

    estimator.local_estimator = RandomForestRegressor()
    estimator.use_stacking = False
    estimator.fit(X, y_true, [xy_vector])
    t3 = t()

    print(t3 - t2, estimator.llocv_score_)
    # 1075.0753190517426 0.8228107533011373
    # 1071.8284921646118 0.8242928357531293

    estimator.local_estimator = LinearRegression()
    estimator.use_stacking = False
    estimator.fit(X, y_true, [xy_vector])
    t4 = t()

    print(t4 - t3, estimator.llocv_score_)


if __name__ == '__main__':
    test_stacking()
