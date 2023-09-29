import matplotlib.pyplot as plt


import numpy as np
from matplotlib import transforms
from sklearn.neighbors import KernelDensity


def plot_ale(fvals, ale, x):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(fvals, ale, zorder=2)

    ax2 = ax1.twinx()
    ax2.hist(x, bins=10, density=False, alpha=0.3, color='gray', zorder=1)

    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.grid(False)

    plt.show()
