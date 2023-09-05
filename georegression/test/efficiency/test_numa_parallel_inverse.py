from time import time

import numpy as np
from numba import njit, prange
# import dpnp as np
# import dpnp.linalg
# import numba_dpex as dpex

# @dpex.dpjit
@njit(parallel=True)
def loop_parallel_inner(iteration_count=1000):
    for i in prange(iteration_count):
        # X = np.identity(100)
        X = np.random.random((100, 100))
        y = np.random.random((100, 1))
        # np.linalg.inv(X)
        # np.linalg.svd(X)

        # augmentation trick to ridge regression
        alpha = 1.0
        A = np.identity(X.shape[1], dtype=X.dtype)

        X_aug = np.vstack((X, np.sqrt(alpha) * A))
        y_aug = np.vstack((y, np.zeros((A.shape[0], 1), dtype=y.dtype)))

        # np.linalg.solve(X, y)
        # np.linalg.solve(X_aug, y_aug)
        np.linalg.lstsq(X_aug, y_aug)


if __name__ == '__main__':
    loop_parallel_inner(1)
    t1 = time()
    loop_parallel_inner(100000)
    t2 = time()
    print(t2 - t1)

    # 128.22677397727966 withouth dpjit