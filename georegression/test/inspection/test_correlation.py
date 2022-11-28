from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

from georegression.test.data import load_HP
from georegression.weight_model import WeightModel

(X, y_true, xy_vector, time) = load_HP()

# TODO: Add labels
labels = np.arange(X.shape[1])


def test_spearman(threshold=1, draw=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 20))
    corr = spearmanr(X).correlation

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    # dist_linkage = hierarchy.single(squareform(distance_matrix))
    dendro = hierarchy.dendrogram(
        dist_linkage, labels=labels, ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro["ivl"]))

    ims = ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]], cmap='PiYG', vmin=-1, vmax=1)
    fig.colorbar(ims, ax=ax2)
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    ax2.set_yticklabels(dendro["ivl"])
    fig.tight_layout()
    if draw:
        plt.show()
    plt.savefig(f'test_corr.png')
    plt.clf()

    cluster_ids = hierarchy.fcluster(dist_linkage, threshold, criterion="distance")
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

    return selected_features


def test_vif(threshold=10, output=False):
    remove_feature_list = []
    while True:
        select_feature_list = np.full(X.shape[1], True)
        select_feature_list[remove_feature_list] = False
        X_selected = X[:, select_feature_list]
        vif_list = [variance_inflation_factor(X_selected, i) for i in range(X_selected.shape[1])]

        # Inf
        # inf_index_list = np.nonzero(np.isinf(vif_list))
        # if inf_index_list[0].shape[0] != 0:
        #     remove_feature_list.append(
        #         np.where(select_feature_list)[inf_index_list[0][0]]
        #     )

        if max(vif_list) < threshold:
            if output:
                print(list(zip(vif_list, labels[select_feature_list])))
            break

        # Remove Max VIF Feature
        vif_index = vif_list.index(max(vif_list))
        remove_feature_list.append(
            np.where(select_feature_list)[0][vif_index]
        )

    return select_feature_list


def test_customer_vif():
    from sklearn.linear_model import LinearRegression
    estimator = LinearRegression()

    feature_index = 12

    estimator.fit(X[:, [*range(feature_index), *range(feature_index + 1, X.shape[1])]], X[:, feature_index])
    r2 = estimator.score(X[:, [*range(feature_index), *range(feature_index + 1, X.shape[1])]], X[:, feature_index])
    VIF = 1 / (1 - r2)
    print(estimator.coef_, r2, VIF)


def test_tree_based_collinear(threshold=0.5, output=False):
    """

    May not work.

    """

    from rfpimp import oob_dependences
    from sklearn.ensemble import RandomForestRegressor
    estimator = RandomForestRegressor(oob_score=True)

    # Not linear regression, not VIF
    # vif_list = 1 / (1 - df_dep.values)

    remove_feature_list = []
    while True:
        select_feature_list = np.full(X.shape[1], True)
        select_feature_list[remove_feature_list] = False
        X_selected = X[:, select_feature_list]
        df_dep = oob_dependences(estimator, pd.DataFrame(X).drop_duplicates(subset=[37, 38])).sort_index()
        score_list = df_dep.values.flatten().tolist()

        if max(score_list) < threshold:
            if output:
                print(list(zip(score_list, labels[select_feature_list])))
            break

        # Remove Max Score Feature
        max_index = score_list.index(max(score_list))
        remove_feature_list.append(
            np.where(select_feature_list)[0][max_index]
        )

    return select_feature_list


def test_select_feature():
    for threshold in np.arange(0.1, 1, 0.1):
        select_features = test_tree_based_collinear(threshold)
        X_selected = X[:, select_features]

        estimator = WeightModel(
            LinearRegression(),
            distance_measure='euclidean',
            kernel_type='bisquare',
            neighbour_count=0.1,

            cache_data=True, cache_estimator=True
        )
        estimator.fit(X_selected, y_true, xy_vector, time)
        print(f'Score {threshold}')
        print(f'Feature Num: {np.nonzero(select_features)[0].shape[0]}')
        print(f'Feature Name: {labels[select_features]}')
        print(estimator.llocv_score_)
        print()
