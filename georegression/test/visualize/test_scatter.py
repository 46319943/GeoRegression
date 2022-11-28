from sklearn.linear_model import LinearRegression

from georegression.test.data import load_HP
from georegression.visualize.pd import partial_cluster
from georegression.weight_model import WeightModel
from georegression.visualize.scatter import scatter_3d

X, y, xy_vector, time = load_HP()


def test_scatter():
    model = WeightModel(
        LinearRegression(),
        distance_measure='euclidean',
        kernel_type='bisquare',
        neighbour_count=0.5,

        cache_data=True, cache_estimator=True
    )

    # Continuous case
    scatter_3d(
        xy_vector[:100], time[:100], y[:100],
        'Title', 'Continuous'
    )

    # Cluster case
    model.fit(X[:100, :10], y[:100], [xy_vector[:100], time[:100]])
    model.partial_dependence()
    feature_distance, feature_cluster_label, distance_matrix, cluster_label = partial_cluster(model.feature_partial_)

    scatter_3d(
        xy_vector[:100], time[:100], cluster_label[:100],
        'Title', 'Cluster', is_cluster=True
    )


if __name__ == '__main__':
    test_scatter()
