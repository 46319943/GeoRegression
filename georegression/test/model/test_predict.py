from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from georegression.test.data import load_HP
from georegression.weight_model import WeightModel

X, y, xy_vector, time = load_HP()


def test_predict():
    model = WeightModel(
        LinearRegression(),
        distance_measure='euclidean',
        kernel_type='bisquare',
        neighbour_count=0.1,

        cache_data=True, cache_estimator=True
    )

    model.fit(X, y, [xy_vector, time])
    prediction_by_weight = model.predict_by_weight(X[:10], [xy_vector[:10], time[:10]])
    prediction_by_fit = model.predict_by_fit(X[:10], [xy_vector[:10], time[:10]])

    # Prediction outperform the local_predict because of the data leak.
    print(
        r2_score(y[:10], model.local_predict_[:10]),
        r2_score(y[:10], prediction_by_weight),
        r2_score(y[:10], prediction_by_fit),
    )


if __name__ == '__main__':
    test_predict()
