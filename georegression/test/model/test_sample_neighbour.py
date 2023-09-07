import numpy as np

from georegression.neighbour_utils import sample_neighbour


def test_sample_neighbour():
    # Generate ramdom weight matrix
    weight_matrix = np.random.rand(100, 100)

    # Set 0 randomly
    weight_matrix[weight_matrix < 0.99] = 0

    # Sample neighbour
    neighbour_matrix_sampled = sample_neighbour(weight_matrix, sample_rate=0.5)

if __name__ == '__main__':
    test_sample_neighbour()
