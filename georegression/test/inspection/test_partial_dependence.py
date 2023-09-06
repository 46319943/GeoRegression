from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from georegression.test.data import load_HP
from georegression.visualize.pd import select_partials
from georegression.weight_model import WeightModel


X, y, xy_vector, time = load_HP()


def test_partial_dependence():
    model = WeightModel(
        LinearRegression(),
        distance_measure='euclidean',
        kernel_type='bisquare',
        neighbour_count=0.5,

        cache_data=True, cache_estimator=True
    )

    model.fit(X[:100, :10], y[:100], [xy_vector[:100], time[:100]])
    model.partial_dependence()
    model.local_ICE()


if __name__ == '__main__':
    test_partial_dependence()
