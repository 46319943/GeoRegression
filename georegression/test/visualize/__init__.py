def get_toy_model():
    from georegression.test.data import load_HP
    X, y, xy_vector, time = load_HP()
    from georegression.weight_model import WeightModel
    from sklearn.linear_model import LinearRegression
    model = WeightModel(
        LinearRegression(),
        distance_measure='euclidean',
        kernel_type='bisquare',
        neighbour_count=0.5,
        cache_data=True, cache_estimator=True
    )
    model.fit(X[:100, :10], y[:100], [xy_vector[:100], time[:100]])
    model.partial_dependence()

    return model
