import numpy as np
from georegression.simulation.simulation_for_fitting import generate_sample, f_square, coef_strong

X, y, points = generate_sample(500, f_square, coef_strong, random_seed=1, plot=True)
X_plus = np.concatenate([X, points], axis=1)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from georegression.stacking_model import StackingWeightModel

distance_measure = "euclidean"
kernel_type = "bisquare"

stacking_neighbour_count=0.3
stacking_neighbour_leave_out_rate=0.1
model = StackingWeightModel(
    DecisionTreeRegressor(splitter="random", max_depth=X.shape[1]),
    # Or use the ExtraTreesRegressor for better predicting performance.
    # ExtraTreesRegressor(n_estimators=10, max_depth=X.shape[1]), 
    distance_measure,
    kernel_type,
    neighbour_count=stacking_neighbour_count,
    neighbour_leave_out_rate=stacking_neighbour_leave_out_rate,
)
model.fit(X_plus, y, [points])
print('STST R2 Score: ', model.llocv_stacking_)

# --- Alternative ---

from sklearn.metrics import r2_score
y_predict = model.stacking_predict_
score = r2_score(y, y_predict)
print(score)


# --- Context ---

X_test, y_test, points_test = generate_sample(500, f_square, coef_strong, random_seed=2, plot=False)
X_test_plus = np.concatenate([X_test, points_test], axis=1)

y_predict = model.predict_by_fit(X_plus, y, [points], X_test_plus, [points_test])

# For weight model:
# y_predict = model.predict_by_fit(X_test_plus, [points_test])

# For predict by weight:
# y_predict = model.predict_by_weight(X_test_plus, [points_test])
score = r2_score(y_test, y_predict)