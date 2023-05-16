def test_pd_sample():
    from georegression.test.visualize import get_toy_model
    model = get_toy_model()

    from georegression.visualize.pd import sample_partial
    sample_partial(
        model.feature_partial_[0], quantile=[0.1, 0.5]
    )

    from georegression.visualize.pd import partial_cluster
    _, cluster_label, _ = partial_cluster(model.feature_partial_[0])
    sample_partial(
        model.feature_partial_[0], sample_size=0.1, cluster_label=cluster_label
    )


if __name__ == '__main__':
    test_pd_sample()
