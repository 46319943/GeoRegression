import dpnp as np

x = np.asarray([1, 2, 3])
print("Array x allocated on the device:", x.device)

y = np.sum(x)

print("Result y is located on the device:", y.device)  # The same device as x
print("Shape of y is:", y.shape)  # 0-dimensional array
print("y =", y)  # Expect 6