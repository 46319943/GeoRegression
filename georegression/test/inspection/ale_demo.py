import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

X = np.random.normal(size=(100, 5))
y = np.random.normal(size=100)

df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

estimator = RandomForestRegressor(n_estimators=100)
estimator.fit(X, y)

def PyALE():
    from PyALE import ale
    ale(df, estimator, ['x1'])


def alibiALE():
    from alibi.explainers import ALE
    def predict_fn(X):
        return estimator.predict(X)

    ale = ALE(predict_fn)
    exp = ale.explain(X)

def ALEPython():
    from alepython import ale_plot
    
    # Plots ALE of feature 'cont' with Monte-Carlo replicas (default : 50).
    ale_plot(model, X_train, "cont", monte_carlo=True)
    

if __name__ == '__main__':
    alibiALE()