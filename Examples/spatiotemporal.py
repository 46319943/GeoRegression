import numpy as np
from georegression.simulation.simulation_for_fitting import generate_sample, f_square, coef_strong

X, y, points = generate_sample(500, f_square, coef_strong, random_seed=1, plot=True)
X_plus = np.concatenate([X, points], axis=1)

from sklearn.ensemble import RandomForestRegressor
from georegression.weight_model import WeightModel

# --- Context ---

times = np.random.randint(0, 10, size=(X.shape[0], 1))
X_plus = np.concatenate([X, points, times], axis=1)

distance_measure = ["euclidean", 'euclidean']
kernel_type = ["bisquare", 'bisquare']

grf_neighbour_count = 0.3

grf_n_estimators=50
model = WeightModel(
    RandomForestRegressor(n_estimators=grf_n_estimators),
    distance_measure,
    kernel_type,
    neighbour_count=grf_neighbour_count,
)
model.fit(X_plus, y, [points, times])