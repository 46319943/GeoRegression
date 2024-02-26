import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from georegression.local_ale import weighted_ale

from georegression.stacking_model import StackingWeightModel
from georegression.simulation.simulation import generate_sample
from georegression.visualize.ale import plot_ale
from georegression.weight_model import WeightModel

X, y, points, _, _ = generate_sample(count=5000, random_seed=1)
X_plus = np.concatenate([X, points], axis=1)


def test_robust_under_various_data():
    """
    TODO: Explain why the improvement becomes significant when there are more points.

    Returns:

    """
    X, y, points, _, _ = generate_sample(count=1000, random_seed=1)
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
    local_estimator = DecisionTreeRegressor(splitter="random", max_depth=1)
    distance_measure = "euclidean"
    kernel_type = "bisquare"

    neighbour_count = 0.05

    model = StackingWeightModel(
        local_estimator,
        distance_measure,
        kernel_type,
        neighbour_count=neighbour_count,
        neighbour_leave_out_rate=0.25,
    )
    model.fit(X, y, [points])
    print('Stacking:', model.llocv_score_, model.llocv_stacking_)

    neighbour_count = 0.03

    model = WeightModel(
        RandomForestRegressor(n_estimators=50),
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

    model = RandomForestRegressor(oob_score=True, n_estimators=3000, n_jobs=-1)
    model.fit(X_plus, y)
    print('RF:', model.oob_score_)

    model = LinearRegression()
    model.fit(X_plus, y)
    print('LR:', model.score(X_plus, y))


def test_GRF():
    distance_measure = "euclidean"
    kernel_type = "bisquare"

    for neighbour_count in [0.01, 0.02, 0.05]:
        model = WeightModel(
            RandomForestRegressor(n_estimators=50),
            distance_measure,
            kernel_type,
            neighbour_count=neighbour_count,
        )
        model.fit(X, y, [points])
        print('GRF:', model.llocv_score_, neighbour_count)


def test_stacking():
    distance_measure = "euclidean"
    kernel_type = "bisquare"
    local_estimator = DecisionTreeRegressor(splitter="random", max_depth=1)

    for neighbour_count in [0.008, 0.01, 0.012]:
        for leave_out_rate in [0.15, 0.25]:
            model = StackingWeightModel(
                local_estimator,
                distance_measure,
                kernel_type,
                neighbour_count=neighbour_count,
                neighbour_leave_out_rate=leave_out_rate,
            )
            model.fit(X, y, [points])
            print('Stacking:', model.llocv_score_,
                  model.llocv_stacking_, neighbour_count, leave_out_rate)

    # Stacking: 0.5335063048239541 0.6881019591277441 0.01 0.25


def draw_graph():
    X, y, points, f, coef = generate_sample(count=5000, random_seed=1)
    X_plus = np.concatenate([X, points], axis=1)
    distance_measure = "euclidean"
    kernel_type = "bisquare"
    neighbour_count = 0.05

    # local_estimator = DecisionTreeRegressor(splitter="random", max_depth=1)
    # local_estimator = DecisionTreeRegressor(splitter="random", max_depth=2)
    # model = StackingWeightModel(
    #     local_estimator,
    #     distance_measure,
    #     kernel_type,
    #     neighbour_count=neighbour_count,
    #     neighbour_leave_out_rate=0.25,
    #     cache_data=True,
    #     cache_estimator=True,
    # )
    # model.fit(X, y, [points])
    # print('Stacking:', model.llocv_score_, model.llocv_stacking_)

    model = WeightModel(
        RandomForestRegressor(n_estimators=50),
        distance_measure,
        kernel_type,
        neighbour_count=neighbour_count,
        cache_data=True,
        cache_estimator=True,
    )
    model.fit(X, y, [points])
    print('GRF:', model.llocv_score_)

    feature_index = 0

    # ale_list = model.local_ALE(feature_index)
    for local_index in range(model.N):
        # fval, ale = ale_list[local_index]

        estimator = model.local_estimator_list[local_index]
        neighbour_mask = model.neighbour_matrix_[local_index]
        neighbour_weight = model.weight_matrix_[local_index][neighbour_mask]
        X_local = model.X[neighbour_mask]
        ale_result = weighted_ale(
            X_local, feature_index, estimator.predict, neighbour_weight)

        fval, ale = ale_result

        x_neighbour = X[model.neighbour_matrix_[local_index], feature_index]
        y_neighbour = y[model.neighbour_matrix_[local_index]]
        weight_neighbour = model.weight_matrix_[
            local_index, model.neighbour_matrix_[local_index]]
        
        # show_function_at_point(f, coef, points[local_index], ax=ax)
        # Get the true marginal effect for function f = x1 * x2.
        x_gird = np.linspace(np.min(x_neighbour), np.max(x_neighbour), 1000)
        x1 = X[local_index, 1]
        x1 = np.tile(x1, 1000)

        # X_grid = np.stack([x_gird, x1], axis=-1)
        # y_grid = f(X_grid, coef, points[local_index])

        from georegression.simulation.simulation import x2_coef
        beta = coef[0](points[local_index])
        x2_average = x2_coef(points[local_index])
        y_grid = beta * 0.5 * ((x2_average + 2) ** 2 -
                               (x2_average - 2) ** 2) * (1 / 4) * x_gird
        
        x1_base = np.empty(500)
        x1_base[:] = np.min(x_neighbour)
        x2_base = np.random.uniform(x2_average - 2, x2_average + 2, 500)
        base_value_real = estimator.predict(np.stack([x1_base, x2_base], axis=-1)).mean()

        diff = ale[0] - base_value_real
        ale = ale - diff

        fig = plot_ale(fval, ale, x_neighbour)
        ax = fig.get_axes()[0]
        scatter = ax.scatter(x_neighbour, y_neighbour, c=weight_neighbour)
        ax.scatter(X[local_index, feature_index], y[local_index], c='red', label='Local point')
        fig.colorbar(scatter, ax=ax, label='Weight')

        ax.plot(x_gird, y_grid, label="Function value")

        ale_result = weighted_ale(X_local, feature_index, estimator.predict, np.ones(X_local.shape[0]))
        fval, ale = ale_result
        diff = ale[0] - base_value_real
        ale = ale - diff
        # ax.plot(fval, ale, label="Neighbour ALE")


        # Add non-weighted ALE plot
        # ale_result = weighted_ale(X_local, feature_index, estimator.predict, np.ones(X_local.shape[0]))
        # Select the X that is in the value range of x_neighbour
        x_global_ale = X[(X[:, feature_index] >= np.min(x_neighbour)) & (X[:, feature_index] <= np.max(x_neighbour))]
        ale_result = weighted_ale(
            x_global_ale, feature_index, estimator.predict, np.ones(x_global_ale.shape[0]))
        fval, ale = ale_result

        x1_base = np.empty(500)
        x1_base[:] = np.min(x_neighbour)
        x2_base = np.random.choice(X[:, 1], 500)
        base_value_real = estimator.predict(np.stack([x1_base, x2_base], axis=-1)).mean()

        diff = ale[0] - base_value_real
        ale = ale - diff

        ax.plot(fval, ale, label="Non-weighted ALE")        

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        handles.append(scatter)
        labels.append('Weight')
        ax.legend(handles, labels)

        plt.show(block=True)

    importance_global = model.importance_score_global()
    print(importance_global)

    importance_local = model.importance_score_local()
    print(importance_local)

    # Normalize the local importance to [0, 1]
    importance_local = (importance_local - importance_local.min()) / \
        (importance_local.max() - importance_local.min())

    # Plot the local importance
    fig, ax = plt.subplots()
    scatter = ax.scatter(points[:, 0], points[:, 1],
                         c=importance_local[:, 0], cmap='viridis')
    fig.colorbar(scatter)
    plt.show()

    # Show the residual across the space.
    residual = model.stacking_predict_ - model.y_sample_
    residual = np.abs(residual)
    fig, ax = plt.subplots()
    # Lower residual values has lower transparency
    scatter = ax.scatter(points[:, 0], points[:, 1],
                         c=residual, alpha=residual / residual.max())
    fig.colorbar(scatter)
    fig.savefig('residual.png')

    fval, ale = model.global_ALE(feature_index)
    fig = plot_ale(fval, ale, X[:, feature_index])
    fig.savefig('ale_global.png')


if __name__ == "__main__":
    # test_robust_under_various_data()
    # test_without_X_plus()
    # test_GRF()
    # test_stacking()

    draw_graph()
