from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from georegression.test.data import load_HP
from georegression.weight_model import WeightModel
from georegression.visualize.ale import plot_ale

X, y, xy_vector, time = load_HP()


def test_ale():
    model = WeightModel(
        # LinearRegression(),
        RandomForestRegressor(n_estimators=10),
        distance_measure='euclidean',
        kernel_type='bisquare',
        neighbour_count=0.1,
        cache_data=True, cache_estimator=True, n_jobs=-1
    )

    model.fit(X[:, -5:], y, [xy_vector, time])
    ale_list = model.local_ALE(0)

    for ale in ale_list:
        fval, ale = ale
        plot_ale(fval, ale)

    print()



if __name__ == '__main__':
    test_ale()
