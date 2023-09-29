from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from georegression.test.data import load_HP
from georegression.weight_model import WeightModel
from georegression.visualize.ale import plot_ale

X, y, xy_vector, time = load_HP()
X = X[:200]
y = y[:200]
xy_vector = xy_vector[:200]
time = time[:200]

def test_ale():
    global X

    model = WeightModel(
        # LinearRegression(),
        RandomForestRegressor(n_estimators=10),
        distance_measure='euclidean',
        kernel_type='bisquare',
        neighbour_count=0.1,
        cache_data=True, cache_estimator=True, n_jobs=1
    )

    X = X[:, -5:]
    model.fit(X, y, [xy_vector, time])
    for i in range(5):
        feature_index = i
        fval, ale, _ = model.global_ALE(feature_index)
        plot_ale(fval, ale, X[:, feature_index])

    print()

if __name__ == '__main__':
    test_ale()
