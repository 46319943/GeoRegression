from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from georegression.test.data.simulation import generate_sample
from georegression.weight_model import WeightModel

X, y, points, coefficients = generate_sample(42)


def main():
    model = WeightModel(
        LinearRegression(),
        distance_measure='euclidean',
        kernel_type='bisquare',
        neighbour_count=0.6,
        cache_data=True, cache_estimator=True
    )

    model.fit(X, y, [points])
    print(
        model.llocv_score_
    )

    model = WeightModel(
        RandomForestRegressor(),
        distance_measure='euclidean',
        kernel_type='bisquare',
        neighbour_count=0.6,
        cache_data=True, cache_estimator=True
    )

    model.fit(X, y, [points])
    print(
        model.llocv_score_
    )

    # Fit random forest on the data and print oob score
    model = RandomForestRegressor(oob_score=True)
    model.fit(X, y)
    print(
        model.oob_score_
    )



if __name__ == '__main__':
    main()
