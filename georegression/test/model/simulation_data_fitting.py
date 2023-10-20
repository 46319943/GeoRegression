import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from georegression.local_ale import weighted_ale

from georegression.stacking_model import StackingWeightModel
from georegression.test.data.simulation import generate_sample
from georegression.visualize.ale import plot_ale
from georegression.weight_model import WeightModel

X, y, points, _ = generate_sample()


def test_nonlinear_spatiotemporal_really_work():
    neighbour_count = 0.4

    model = WeightModel(
        LinearRegression(),
        distance_measure="euclidean",
        kernel_type="bisquare",
        neighbour_count=neighbour_count,
        cache_data=True,
        cache_estimator=True,
    )

    model.fit(X, y, [points])
    print(model.llocv_score_)

    model = WeightModel(
        RandomForestRegressor(n_estimators=50),
        distance_measure="euclidean",
        kernel_type="bisquare",
        neighbour_count=neighbour_count,
        cache_data=True,
        cache_estimator=True,
    )

    model.fit(X, y, [points])
    print(model.llocv_score_)

    # Fit random forest on the data and print oob score
    model = RandomForestRegressor(oob_score=True)
    model.fit(X, y)
    print(model.oob_score_)


def test_robust_under_various_data():
    """
    TODO: Explain why the improvement becomes significant when there are more points.

    Returns:

    """
    X, y, points, _ = generate_sample(count=1000, random_seed=1)
    X_plus = np.concatenate([X, points], axis=1)

    local_estimator = DecisionTreeRegressor(splitter="random", max_depth=1)
    distance_measure = "euclidean"
    kernel_type = "bisquare"

    neighbour_count = 0.03

    model = StackingWeightModel(
        local_estimator,
        distance_measure,
        kernel_type,
        neighbour_count=neighbour_count,
        neighbour_leave_out_rate=0.15,
    )
    model.fit(X_plus, y, [points])
    print('Stacking:', model.llocv_score_, model.llocv_stacking_)

    model = WeightModel(
        RandomForestRegressor(n_estimators=200),
        distance_measure,
        kernel_type,
        neighbour_count=neighbour_count,
    )
    model.fit(X_plus, y, [points])
    print('GRF:', model.llocv_score_)

    model = WeightModel(
        LinearRegression(),
        distance_measure,
        kernel_type,
        neighbour_count=neighbour_count,
    )
    model.fit(X_plus, y, [points])
    print('GWR:', model.llocv_score_)

    model = RandomForestRegressor(oob_score=True, n_estimators=5500, n_jobs=-1)
    model.fit(X_plus, y)
    print('RF:', model.oob_score_)

    model = LinearRegression()
    model.fit(X_plus, y)
    print('LR:', model.score(X_plus, y))

    """
    count=500
    0.7813469046663418
    0.7334095600009363
    0.5015610109930759
    
    count=5000
    0.8692464077285508
    0.7648331307574766
    0.5087431918278111
    
    count=5000
    neighbour_count = 0.1
    0.94041696882913
    0.9466926082507984
    0.5182052654589111
    """


def test_without_X_plus():
    X, y, points, _ = generate_sample(count=5000, random_seed=1)
    X_plus = np.concatenate([X, points], axis=1)

    local_estimator = DecisionTreeRegressor(splitter="random", max_depth=1)
    distance_measure = "euclidean"
    kernel_type = "bisquare"

    neighbour_count = 0.03

    model = StackingWeightModel(
        local_estimator,
        distance_measure,
        kernel_type,
        neighbour_count=neighbour_count,
        neighbour_leave_out_rate=0.15,
    )
    model.fit(X, y, [points])
    print('Stacking:', model.llocv_score_, model.llocv_stacking_)

    model = WeightModel(
        RandomForestRegressor(n_estimators=200),
        distance_measure,
        kernel_type,
        neighbour_count=neighbour_count,
    )
    model.fit(X, y, [points])
    print('GRF:', model.llocv_score_)

    model = WeightModel(
        LinearRegression(),
        distance_measure,
        kernel_type,
        neighbour_count=neighbour_count,
    )
    model.fit(X, y, [points])
    print('GWR:', model.llocv_score_)

    model = RandomForestRegressor(oob_score=True, n_estimators=5500, n_jobs=-1)
    model.fit(X_plus, y)
    print('RF:', model.oob_score_)

    model = LinearRegression()
    model.fit(X_plus, y)
    print('LR:', model.score(X_plus, y))

def draw_graph():
    X, y, points, _ = generate_sample(count=3000, random_seed=1)
    X_plus = np.concatenate([X, points], axis=1)

    local_estimator = DecisionTreeRegressor(splitter="random", max_depth=1)
    distance_measure = "euclidean"
    kernel_type = "bisquare"

    neighbour_count = 0.015

    model = StackingWeightModel(
        local_estimator,
        distance_measure,
        kernel_type,
        neighbour_count=neighbour_count,
        neighbour_leave_out_rate=0.15,
        cache_data=True
    )
    model.fit(X, y, [points])
    print('Stacking:', model.llocv_score_, model.llocv_stacking_)

    # Show the residual across the space.
    residual = model.stacking_predict_ - model.y_sample_
    residual = np.abs(residual)
    plt.figure()
    # Lower residual values has lower transparency
    plt.scatter(points[:, 0], points[:, 1], c=residual, alpha=residual / residual.max())
    plt.colorbar()
    plt.show()

    feature_index = 0
    fval, ale = model.global_ALE(feature_index)
    fig = plot_ale(fval, ale, X[:, feature_index])
    fig.show()

    # ale_list = model.local_ALE(feature_index)

    for local_index in range(model.N):
        # fval, ale = ale_list[local_index]

        estimator = model.local_estimator_list[local_index]
        neighbour_mask = model.neighbour_matrix_[local_index]
        neighbour_weight = model.weight_matrix_[local_index][neighbour_mask]
        X_local = model.X[neighbour_mask]
        ale_result = weighted_ale(X_local, feature_index, estimator.predict, neighbour_weight)

        fval, ale = ale_result

        x_neighbour = X[model.neighbour_matrix_[local_index], feature_index]
        y_neighbour = y[model.neighbour_matrix_[local_index]]
        weight_neighbour = model.weight_matrix_[local_index, model.neighbour_matrix_[local_index]]

        fig = plot_ale(fval, ale, x_neighbour)

        ax = fig.get_axes()[0]
        scatter = ax.scatter(x_neighbour, y_neighbour, c=weight_neighbour)
        ax.scatter(X[local_index, feature_index], y[local_index], c='red')
        fig.colorbar(scatter, ax=ax, label='Weight') 

        fig.show()

        # Plot the neighbour, with the color as the weight
        plt.figure()
        plt.scatter(x_neighbour, y_neighbour, c=weight_neighbour)
        plt.colorbar()
        # Plot the local point
        plt.scatter(X[local_index, feature_index], y[local_index], c='red')
        plt.show()

        y_predict = estimator.predict(X_local)
        y_predict_local = y_predict[np.argmax(weight_neighbour)]

        plt.figure()
        plt.scatter(x_neighbour, y_predict, c=weight_neighbour)
        plt.colorbar()
        # Get the prediction of the local point
        plt.scatter(X[local_index, feature_index], y_predict_local, c='red')
        plt.show(block=True)



    importance_global = model.importance_score_global()
    print(importance_global)

    importance_local = model.importance_score_local()
    print(importance_local)

    # Plot the local importance
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], c=importance_local)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    # test_nonlinear_spatiotemporal_really_work()
    # test_robust_under_various_data()
    # test_without_X_plus()

    draw_graph()