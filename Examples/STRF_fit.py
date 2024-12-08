import numpy as np
from georegression.simulation.simulation_for_fitting import generate_sample, f_square, coef_strong

X, y, points = generate_sample(500, f_square, coef_strong, random_seed=1, plot=True)
X_plus = np.concatenate([X, points], axis=1)

# --- Context ---

from sklearn.ensemble import RandomForestRegressor
from georegression.weight_model import WeightModel

distance_measure = "euclidean"
kernel_type = "bisquare"

grf_neighbour_count=0.3
grf_n_estimators=50
model = WeightModel(
    RandomForestRegressor(n_estimators=grf_n_estimators),
    distance_measure,
    kernel_type,
    neighbour_count=grf_neighbour_count,
)
model.fit(X_plus, y, [points])
print('STRF R2 Score: ', model.llocv_score_)

# --- Alternative ---

from sklearn.metrics import r2_score
y_predict = model.local_predict_
score = r2_score(y, y_predict)
print(score)
