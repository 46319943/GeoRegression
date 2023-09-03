import numpy as np
from numba import jit, njit
from sklearn.linear_model import Ridge
from time import  time

@jit(forceobj=True, looplift=True)
def lr():
    X = np.random.random((100, 100))
    y = np.random.random((100, 1))
    estimator = Ridge(0.1)
    estimator.fit(X, y)

    return estimator
@jit(forceobj=True, looplift=True)
def loop_lr():
    for i in range(1000):
        lr()

if __name__ == '__main__':
    loop_lr()

    t1 = time()
    loop_lr()
    t2 = time()
    print(t2 - t1)
