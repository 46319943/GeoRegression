import numpy as np
from scipy import sparse

M = sparse.csr_matrix([[0, 0, 0], [2, 0, 3], [4, 4, 5]])
ML = M.tolil()

for d, r in enumerate(zip(ML.data, ML.rows)):
    # d,r are lists
    dr = np.array([d, r])
    print(dr)
