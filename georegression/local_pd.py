import numpy as np
from sklearn.inspection import partial_dependence


def local_partial_dependence(local_estimator, X, weight):
    """

    Args:
        local_estimator ():
        X ():
        weight ():

    Returns:

    """
    # TODO: More detailed reason to justify the weighted partial dependence.
    # Care more on the local range.
    # Unweighted points will dominate the tendency which may not be the interested one.
    # Only calculate the local ICE?
    # Better explanation in ALE: the adverse consequences of extrapolation in PD plots
    # Ref: Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models


    feature_count = X.shape[1]

    # Select X to speed up calculation
    select_mask = weight != 0
    X = X[select_mask]
    weight = weight[select_mask]

    # Partial result of each features
    feature_list = []

    for feature_index in range(feature_count):
        pdp = partial_dependence(
            local_estimator,
            X,
            [feature_index],
            kind='both'
        )

        # Must get individual partial dependence to weight the result
        # Weight: Performance Weight. The point with more weight performance better in the model.
        # So average the partial performance according to the weight.
        individual = pdp['individual'][0]
        values = pdp['values'][0]
        weight_average = np.average(individual, axis=0, weights=weight)

        # TODO: Pack the result
        feature_list.append({
            'x': values,
            'pd': weight_average
        })

    return feature_list

# TODO: Add local ICE
