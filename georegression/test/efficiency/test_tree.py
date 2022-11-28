import time

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from gwrf.test.data import TOD
from joblib.parallel import Parallel, delayed


def test_tree_efficiency():
    """
    对于某局部模型，给定局部数据X
    Stacking：训练单个简单的树，对X个邻接的X进行预测。
    训练时间train*data(X)，预测时间predict*data(X)*number(X)。实际中预测的X有重复，应小于这个值。
    线性Stacking时间Inverse(X*X) X个样本，X个邻接提供的X个特征。

    RandomForest：训练n个完全树，进行平均。
    训练时间train*data(X)*number(n)

    Returns:

    """
    neighbour_count = 5000
    feature_count = 30
    X = np.random.random((neighbour_count, feature_count))
    X_bulk = np.concatenate([X] * neighbour_count, axis=0)
    y = np.random.random(neighbour_count)

    # Wamp-up

    tree = DecisionTreeRegressor()
    tree.fit(X, y)
    tree.predict(X)

    forest = RandomForestRegressor()
    forest.fit(X, y)

    t1 = time.time()

    tree = DecisionTreeRegressor(max_depth=2, splitter='random')
    tree.fit(X, y)

    t2 = time.time()

    # Predict in atomic. But not completely atomic.
    for i in range(neighbour_count):
        tree.predict(X)

    t3 = time.time()

    # Bulk is better for moderate amount of data. (100~500)^2. Degrade when come to 1000^2.
    # https://scikit-learn.org/0.15/modules/computational_performance.html
    # https://scikit-learn.org/0.15/auto_examples/applications/plot_prediction_latency.html#example-applications-plot-prediction-latency-py
    tree.predict(X_bulk)

    t4 = time.time()

    forest = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    forest.fit(X, y)

    t5 = time.time()

    print()
    print(t2 - t1, t3 - t2, t4 - t3, t5 - t4)
    # neighbour = 100 0.0 0.00400090217590332 0.0009999275207519531 0.09102034568786621
    # neighbour = 500 0.0 0.02702188491821289 0.016988277435302734 0.12402749061584473
    # neighbour = 1000 0.0009999275207519531 0.06401467323303223 0.06501483917236328 0.1708080768585205
    # neighbour = 2000 0.0010001659393310547 0.18304109573364258 0.2760617733001709 0.3486180305480957
    # neighbour = 5000 0.003000497817993164 0.8315958976745605 3.4696438312530518 1.164928913116455


def fit_estimator(estimator, X, y):
    estimator.fit(X, y)
    return estimator


def estimators_predict(estimator_list, X):
    result_list = []
    for estimator in estimator_list:
        result_list.append(
            estimator.predict_by_weight(X)
        )
    return result_list


def estimators_parallel_predict(estimator_list, X):
    return Parallel(-1)(
        delayed(estimator_predict)(
            estimator, X
        )
        for estimator in estimator_list
    )


def estimator_predict(estimator, X):
    return estimator.predict_by_weight(X)


def estimator_predict_n_time(estimator, X, times):
    predict_result = []
    for i in range(times):
        predict_result.append(
            estimator.predict_by_weight(X)
        )
    return predict_result


def test_in_joblib():
    # 邻接数量
    neighbour_count = 100
    # 局部模型数量
    estimator_count = 100

    feature_count = 30
    X = np.random.random((neighbour_count, feature_count))
    X_bulk = np.concatenate([X] * neighbour_count, axis=0)
    y = np.random.random(neighbour_count)

    parallel = Parallel(-1)

    # Warm-up
    parallel([
        delayed(fit_estimator)(
            DecisionTreeRegressor(), X, y
        )
        for i in range(64)
    ])
    parallel([
        delayed(fit_estimator)(
            RandomForestRegressor(), X, y
        )
        for i in range(64)
    ])

    print('Tree fitting starts.')
    t1 = time.time()

    tree_list = parallel([
        delayed(fit_estimator)(
            DecisionTreeRegressor(max_depth=2, splitter='random'), X, y
        )
        for i in range(estimator_count)
    ])

    print('Prediction Starts.')
    t2 = time.time()

    # Really slow. Resource not fully used.

    # Only get the theoretical time when neighbour count equals estimator count.
    # i.e. tree list equals neighbour estimator list.
    tree_result = parallel([
        delayed(estimators_predict)(
            tree_list, X
        )
        for i in range(estimator_count)
    ])

    t3 = time.time()

    tree_result = parallel(
        delayed(estimator_predict_n_time)(
            estimator, X, neighbour_count
        )
        for estimator in tree_list
    )

    t4 = time.time()

    for estimator in tree_list:
        for i in range(neighbour_count):
            estimator.predict_by_weight(X)

    t5 = time.time()

    for estimator in tree_list:
        estimator.predict_by_weight(X_bulk)

    t6 = time.time()

    tree_result = parallel(
        delayed(estimator_predict)(
            estimator, X_bulk
        )
        for estimator in tree_list
    )

    t7 = time.time()

    split_num = 6
    tree_result = parallel(
        delayed(estimators_predict)(
            [tree_list[i] for i in split_index], X_bulk
        )
        for split_index in np.array_split(np.arange(estimator_count), split_num)
    )

    print('Prediction Ends.')
    t8 = time.time()

    forest_list = parallel([
        delayed(fit_estimator)(
            RandomForestRegressor(n_estimators=100, n_jobs=-1), X, y
        )
        for i in range(estimator_count)
    ])

    t9 = time.time()

    print()
    print(t2 - t1)
    print(t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6, t8 - t7)
    print(t9 - t8)
    # neighbour = estimator = 100
    # 0.03500962257385254
    # 0.65926194190979 0.09602212905883789 0.3746037483215332 0.05401206016540527 0.17014646530151367 0.05001258850097656
    # 1.7216594219207764


if __name__ == '__main__':
    test_in_joblib()
