# GeoRegression
> A geospatial framework for performing non-linear regression, designed to effectively model complex spatial relationships.

<p align="center">
  <img src="Images/icon.png" width="200">
</p>

<p align="center">
  <a href="https://github.com/yqx-github/GeoRegression/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://pypi.org/project/georegression/"><img src="https://img.shields.io/pypi/v/georegression" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/release/python-370/"><img src="https://img.shields.io/badge/python-3.7+-blue.svg" alt="Python"></a>
</p>

This Python package offers a robust framework for regression modeling on geospatial data, addressing the challenge of spatial non-stationarity by integrating spatial information directly into the modeling process. Built on this framework are two advanced methods: the *SpatioTemporal Random Forest* (STRF) and the *SpatioTemporal Stacking Tree* (STST), which leverage spatial and temporal patterns to enhance predictive accuracy.

![Illustration for STRF and STST](Images/pipeline.png)

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Data Preparation](#data-preparation)
  - [Shared Parameter Explanation](#shared-parameter-explanation)
  - [SpatioTemporal Random Forest (STRF)](#spatiotemporal-random-forest-strf)
  - [SpatioTemporal Stacking Tree (STST)](#spatiotemporal-stacking-tree-stst)
  - [GWR / GTWR](#gwr--gtwr)
  - [Prediction](#prediction)
  - [SpatioTemporal Dimension](#spatiotemporal-dimension)
  - [Posterior Inspection Tools](#posterior-inspection-tools)
    - [Feature Importance Analysis](#feature-importance-analysis)
    - [SpatioTemporal (Local) Accumulated Local Effects (STALE) Plots](#spatiotemporal-local-accumulated-local-effects-stale-plots)
- [Citation](#citation)


# Installation

Python with version >= 3.7 is required.

```bash
pip install georegression
```

# Quick Start
- The full example can be found in the `Examples` folder.

## Data Preparation
- Use the provided function to generate the sample data with spatial non-stationarity.
```python
import numpy as np
from georegression.simulation.simulation_for_fitting import generate_sample, f_square, coef_strong

X, y, points = generate_sample(500, f_square, coef_strong, random_seed=1, plot=True)
X_plus = np.concatenate([X, points], axis=1)
```

## Shared Parameter Explanation
Several parameters are shared across different model implementations and are used to construct weight matrices for both spatial and spatiotemporal dimensions:

- `kernel_type`: Determines the kernel function used for spatial weighting. Accepts standard kernel types:
  - `'bisquare'`: A commonly used kernel that provides smooth distance-based weighting
  - `'gaussian'`: Gaussian kernel for distance-based weighting

- `neighbour_count`: Controls the adaptive kernel bandwidth for spatial weighting:
  - Must be a decimal between 0 and 1 (e.g., 0.3)
  - Uses an adaptive kernel bandwidth equal to the distance to the specified percentage of nearest neighbors

- `bandwidth`: Optional parameter for fixed kernel bandwidth:
  - If an integer value is provided, uses this fixed distance as the kernel bandwidth
  - This bandwidth will be applied uniformly to all data points
  - When specified, this takes precedence over `neighbour_count`

## SpatioTemporal Random Forest (STRF)
- The `WeightModel` class provides the basic weighted framework for regression.
- In the weighted framework, each local models do not see the y value of the target location, therefore, the prediction of each local model is the prediction of the whole model.

```python
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

```

## SpatioTemporal Stacking Tree (STST)
- The `StackingWeightModel` class provides the weighted stacking framework for regression.
- In the weighted stacking framework, each local models do not see the y value of the target location, therefore, the prediction of each local model is the prediction of the whole model.

```python
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
```

## GWR / GTWR
```python
from sklearn.linear_model import LinearRegression
from georegression.weight_model import WeightModel

distance_measure = "euclidean"
kernel_type = "bisquare"

gwr_neighbour_count=0.2
model = WeightModel(
    LinearRegression(),
    distance_measure,
    kernel_type,
    neighbour_count=gwr_neighbour_count,
)
model.fit(X_plus, y, [points])

print('GWR R2 Score: ', model.llocv_score_)

# --- Alternative ---

from sklearn.metrics import r2_score
y_predict = model.local_predict_
score = r2_score(y, y_predict)
print(score)
```

## Prediction
- Although in the weighted framework, the prediction of each local model is the prediction of the whole model, two methods are provided for making prediction for the new data:
    - `predict_by_fit`: Fit new local model for prediction data using the training data to make prediction.
    - `predict_by_weight`: Predict using local estimators and weight the local predictions using the weight matrix that calculated by using training locations as source and prediction locations as target.

```python
X_test, y_test, points_test = generate_sample(500, f_square, coef_strong, random_seed=2, plot=False)
X_test_plus = np.concatenate([X_test, points_test], axis=1)

y_predict = model.predict_by_fit(X_plus, y, [points], X_test_plus, [points_test])

# For weight model:
# y_predict = model.predict_by_fit(X_test_plus, [points_test])

# For predict by weight:
# y_predict = model.predict_by_weight(X_test_plus, [points_test])
score = r2_score(y_test, y_predict)
print(score)
```

## SpatioTemporal Dimension
- To use more than one dimension of spatial information, just add the new dimension to the input data.

```python
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
```

## Posterior Inspection Tools
GeoRegression provides powerful tools for model interpretation and analysis after fitting. Here are two key features:

### Feature Importance Analysis
You can analyze both global and local feature importance to understand how different features contribute to predictions across space:

```python
from georegression.weight_model import WeightModel
from sklearn.ensemble import RandomForestRegressor

# Fit the model
model = WeightModel(
    RandomForestRegressor(n_estimators=50),
    distance_measure="euclidean",
    kernel_type="bisquare",
    neighbour_count=0.02
)
model.fit(X, y, [points])

# Get global feature importance
importance_global = model.importance_score_global()
print("Global Importance Score: ", importance_global)

# Get local feature importance
importance_local = model.importance_score_local()

# Visualize local importance for each feature
import matplotlib.pyplot as plt

for i in range(importance_local.shape[1]):
    plt.figure()
    scatter = plt.scatter(
        points[:, 0], points[:, 1], 
        c=importance_local[:, i], 
        cmap="viridis"
    )
    plt.colorbar(scatter)
    plt.title(f"Local Importance of Feature {i}")
    plt.show()
```

Example visualization of local feature importance:

<p align="center">
  <img src="Images/Local_importance_0.png" width="600">
  <br>
  <em>Local importance visualization showing spatial variation in feature influence</em>
</p>

### SpatioTemporal (Local) Accumulated Local Effects (STALE) Plots
STALE plots help understand how features affect predictions locally:

```python
from georegression.local_ale import weighted_ale
from georegression.visualize.ale import plot_ale

# For a specific location (local_index)
feature_index = 0  # Feature to analyze
local_index = 0    # Location to analyze

# Get local estimator and data
estimator = model.local_estimator_list[local_index]
neighbour_mask = model.neighbour_matrix_[local_index]
neighbour_weight = model.weight_matrix_[local_index][neighbour_mask]
X_local = model.X[neighbour_mask]

# Calculate ALE
ale_result = weighted_ale(
    X_local, 
    feature_index, 
    estimator.predict, 
    neighbour_weight
)
fval, ale = ale_result

# Plot ALE with weighted observations
x_neighbour = X[model.neighbour_matrix_[local_index], feature_index]
y_neighbour = y[model.neighbour_matrix_[local_index]]
weight_neighbour = model.weight_matrix_[local_index, model.neighbour_matrix_[local_index]]

fig = plot_ale(fval, ale, x_neighbour)
plt.show()
```

Example STALE plot:

<p align="center">
  <img src="Images/STALE_20.png" width="600">
  <br>
  <em>STALE plot showing the local accumulated effects of a feature at a specific location</em>
</p>

These tools provide insights into:
- How different features influence predictions globally and locally
- How feature effects vary across space
- The strength and nature of spatial relationships in your data

# Citation
If you find this package useful in your research, please consider citing:
- Luo, Y., & Su, S. (2025). SpatioTemporal Random Forest and SpatioTemporal Stacking Tree: A novel spatially explicit ensemble learning approach to modeling non-linearity in spatiotemporal non-stationarity. International Journal of Applied Earth Observation and Geoinformation, 136, 104315. https://doi.org/10.1016/j.jag.2024.104315
```
@article{luo_spatiotemporal_2025,
	title = {{SpatioTemporal} {Random} {Forest} and {SpatioTemporal} {Stacking} {Tree}: {A} novel spatially explicit ensemble learning approach to modeling non-linearity in spatiotemporal non-stationarity},
	volume = {136},
	issn = {1569-8432},
	shorttitle = {{SpatioTemporal} {Random} {Forest} and {SpatioTemporal} {Stacking} {Tree}},
	url = {https://www.sciencedirect.com/science/article/pii/S1569843224006733},
	doi = {10.1016/j.jag.2024.104315},
	urldate = {2024-12-30},
	journal = {International Journal of Applied Earth Observation and Geoinformation},
	author = {Luo, Yun and Su, Shiliang},
	month = feb,
	year = {2025},
	keywords = {Ensemble learning, Machine learning, Nonlinearity, Spatially explicit modeling, Spatiotemporal non-stationarity, Spatiotemporal random forest, Spatiotemporal stacking tree},
	pages = {104315},
}
```

