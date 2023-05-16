from georegression.test.data import load_HP
from georegression.test.visualize import get_toy_model
from georegression.visualize.pd import partials_plot_3d, features_partial_cluster, partial_plot_2d, \
    partial_compound_plot, choose_cluster_typical

model = get_toy_model()
features_embedding, features_cluster_label, _ = features_partial_cluster(model.feature_partial_)

X, y, xy_vector, time = load_HP()


def test_compound_plot():
    partial_figs, embedding_figs, cluster_figs, compass_figs = partial_compound_plot(
        xy_vector[:100], time[:100], model.feature_partial_,
        features_embedding, features_cluster_label,
    )


def test_pd_2d_plot():
    # cluster_typical = choose_cluster_typical(cluster_embedding, cluster_label)
    # partial_plot_2d(
    #     model.feature_partial_, cluster_label, cluster_typical,
    #     alpha_range=[0.1, 1], width_range=[0.5, 3], scale_power=1.5
    # )

    cluster_typical = [
        choose_cluster_typical(embedding, cluster)
        for embedding, cluster in zip(features_embedding, features_cluster_label)
    ]
    partial_plot_2d(
        model.feature_partial_, features_cluster_label, cluster_typical,
        alpha_range=[0.3, 1], width_range=[0.5, 3], scale_power=1.5
    )


def test_pd_3d_plot():
    partials_plot_3d(
        model.feature_partial_, model.coordinate_vector_list[1], cluster_labels=features_cluster_label,
        # quantile=[0, 0.2, 0.8, 1],
    )


if __name__ == '__main__':
    test_pd_2d_plot()
    pass
