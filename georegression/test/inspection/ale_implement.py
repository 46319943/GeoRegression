import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from georegression.local_ale import weighted_ale

X = np.random.normal(size=(100, 5))
y = np.random.normal(size=100)

df = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4", "x5"])

estimator = RandomForestRegressor(n_estimators=100)
estimator.fit(X, y)


def test_weighted_ale():
    weighted_ale(X, 0, estimator.predict, weights=np.random.random(size=100))

if __name__ == "__main__":
    test_weighted_ale()
