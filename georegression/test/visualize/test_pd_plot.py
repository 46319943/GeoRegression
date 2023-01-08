from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from georegression.test.data import load_HP
from georegression.visualize.pd import select_partial, partial_plot_3d, partial_cluster, partial_plot_2d, \
    partial_compound_plot
from georegression.weight_model import WeightModel

X, y, xy_vector, time = load_HP()


def test_pd_plot():
    model = WeightModel(
        # LinearRegression(),
        RandomForestRegressor(),
        distance_measure='euclidean',
        kernel_type='bisquare',
        neighbour_count=0.5,

        cache_data=True, cache_estimator=True
    )

    model.fit(X[:100, :10], y[:100], [xy_vector[:100], time[:100]])
    model.partial_dependence()

    feature_embedding, feature_cluster_label, cluster_embedding, cluster_label = partial_cluster(
        model.feature_partial_)

    partial_compound_plot(
        xy_vector[:100], time[:100], model.feature_partial_,
        feature_embedding, feature_cluster_label,
        cluster_embedding, cluster_label
    )
    return

    partial_plot_3d(
        model.feature_partial_, model.coordinate_vector_list[1], cluster_vector=feature_cluster_label,
        # quantile=[0, 0.2, 0.8, 1],
    )
    partial_plot_2d(model.feature_partial_, model.coordinate_vector_list[1], cluster_vector=cluster_label)

    model.local_ICE()
    feature_distance, feature_cluster_label, distance_matrix, cluster_label = partial_cluster(
        xy_vector[:100], time[:100], model.feature_ice_
    )
    partial_plot_3d(
        model.feature_ice_, model.coordinate_vector_list[1], cluster_vector=cluster_label,
        # quantile=[0, 0.2, 0.8, 1],
        is_ICE=True
    )


if __name__ == '__main__':
    test_pd_plot()
