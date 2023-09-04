from numba import njit, prange
from numba.typed import List
import numpy as np


@njit()
def test_list_stack(i, array_to_be_stacked):
    shape = (i,) + array_to_be_stacked.shape
    list_of_array = [array_to_be_stacked] * i
    stacked_array = np.empty(shape)
    for j in prange(i):
        stacked_array[j] = list_of_array[j]
    return stacked_array


@njit()
def stack(list_of_array):
    shape = (len(list_of_array),) + list_of_array[0].shape
    stacked_array = np.empty(shape)
    for j in prange(len(list_of_array)):
        stacked_array[j] = list_of_array[j]
    return stacked_array

if __name__ == "__main__":
    test_list_stack(10, np.ones((2, 3)))

    # Note that you have to use typed list provided by numba here.
    typed_list = List()
    [typed_list.append(np.ones((2, 3))) for _ in range(10)]
    stacked = stack(typed_list)
    print(stacked.shape)
    print(stacked)
