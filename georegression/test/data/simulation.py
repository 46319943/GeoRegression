"""
Generate simulated data for testing purposes.
"""

import random
import time

def radial_coefficient(x, y, z):
    """Return a coefficient for a radial function."""
    return 1 / (x ** 2 + y ** 2 + z ** 2 + 1)