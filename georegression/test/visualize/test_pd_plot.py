import unittest

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from georegression.test.data import load_HP
from georegression.visualize.pd import select_partial, partial_plot_3d, partial_cluster, partial_plot_2d, \
    partial_compound_plot, choose_cluster_typical
from georegression.weight_model import WeightModel

X, y, xy_vector, time = load_HP()


class TestCalculations(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
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

        cls.model = model

        cls.partial_cluster_result = partial_cluster(
            model.feature_partial_)

    def test_compound_plot(self):
        model = self.model
        feature_embedding, feature_cluster_label, cluster_embedding, cluster_label = self.partial_cluster_result

        partial_compound_plot(
            xy_vector[:100], time[:100], model.feature_partial_,
            feature_embedding, feature_cluster_label,
            cluster_embedding, cluster_label
        )

    def test_pd_2d_plot(self):
        model = self.model
        feature_embedding, feature_cluster_label, cluster_embedding, cluster_label = self.partial_cluster_result

        # cluster_typical = choose_cluster_typical(cluster_embedding, cluster_label)
        # partial_plot_2d(
        #     model.feature_partial_, cluster_label, cluster_typical,
        #     alpha_range=[0.1, 1], width_range=[0.5, 3], scale_power=1.5
        # )

        cluster_typical = [
            choose_cluster_typical(embedding, cluster)
            for embedding, cluster in zip(feature_embedding, feature_cluster_label)
        ]
        partial_plot_2d(
            model.feature_partial_, feature_cluster_label, cluster_typical,
            alpha_range=[0.3, 1], width_range=[0.5, 3], scale_power=1.5
        )

    def test_pd_3d_plot(self):
        model = self.model
        feature_embedding, feature_cluster_label, cluster_embedding, cluster_label = self.partial_cluster_result

        partial_plot_3d(
            model.feature_partial_, model.coordinate_vector_list[1], cluster_vector=feature_cluster_label,
            # quantile=[0, 0.2, 0.8, 1],
        )

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
    TestCalculations().test_pd_2d_plot()
    pass
