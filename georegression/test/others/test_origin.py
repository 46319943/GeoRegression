import numpy as np
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

    return estimator.oob_prediction_


def test_ols():
    from sklearn.linear_model import LinearRegression
    e = LinearRegression()
    e.fit(X, y_true)
    y_predict = e.predict(X)
    r2 = r2_score(y_true, y_predict)
    print(r2)

    MAPE = mean_absolute_percentage_error(y_true, y_predict)
    print(MAPE)
