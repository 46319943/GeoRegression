from sklearn.linear_model import LinearRegression

from georegression.test.data import load_HP
from georegression.visualize.importance import global_importance_plot
from georegression.weight_model import WeightModel

X, y, xy_vector, time = load_HP()


def test_importance_plot():
    model = WeightModel(
        LinearRegression(),
        distance_measure='euclidean',
        kernel_type='bisquare',
        neighbour_count=0.1,

        cache_data=True, cache_estimator=True
    )

    # For non-linear interaction test.
    # model.local_estimator = RandomForestRegressor()

    model.fit(X[:, :5], y, [xy_vector, time])
    is_global = model.importance_score_global()
    global_importance_plot(model.permutation_score_decrease_)

    # is_interaction = model.interaction_score_global()


if __name__ == '__main__':
    test_importance_plot()
