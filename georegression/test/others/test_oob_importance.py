from sklearn.metrics import *

from georegression.test.data import load_HP

(X, y_true, xy_vector, time) = load_HP()


def test_rf():
    from sklearn.ensemble import RandomForestRegressor

    print('RandomForest Model')
    estimator = RandomForestRegressor(n_estimators=1000, n_jobs=-1, oob_score=True)
    estimator.fit(X, y_true)
    print(estimator.oob_score_)

    MAPE = mean_absolute_percentage_error(y_true, estimator.oob_prediction_)
    print(MAPE)

    return estimator.predict(estimator.oob_prediction_)

