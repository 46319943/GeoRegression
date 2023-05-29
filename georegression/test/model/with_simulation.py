from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from georegression.test.data.simulation import generate_sample
from georegression.weight_model import WeightModel

X, y, points, coef1, coef2 = generate_sample()
X = X.reshape(-1, 1)


def main():
    model = WeightModel(
        LinearRegression(),
        distance_measure='euclidean',
        kernel_type='bisquare',
        neighbour_count=0.9,
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
        neighbour_count=0.9,
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
