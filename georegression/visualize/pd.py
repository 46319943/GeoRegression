import math
from os.path import join
from pathlib import Path

import numpy as np
from matplotlib import cm, pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.inspection import PartialDependenceDisplay

import plotly.graph_objects as go

from georegression.visualize.scatter import scatter_3d
from georegression.visualize.utils import vector_to_color

from georegression.visualize import folder


def select_partial(feature_partial, sample_size=None, quantile=None):
    """

    Args:
        feature_partial (np.ndarray): Shape(Feature, N, 2)
        sample_size ():
        quantile ():

    Returns: Selection indices. Bool array with shape(Feature, N)

    """

    N = feature_partial.shape[1]
    feature_count = feature_partial.shape[0]

    if sample_size is None and quantile is None:
        raise Exception('Both selection parameter is None.')
    if sample_size is not None and quantile is not None:
        raise Exception('Choose one selection parameter.')

    # Select by sample
    if sample_size is not None:
        # Proportional sample.
        if isinstance(sample_size, float):
            sample_size = int(sample_size * N)

        # random state?
        sample_indices = np.random.choice(N, sample_size, replace=False)

        selection_matrix = np.zeros(N, dtype=bool)
        selection_matrix[sample_indices] = True
        selection_matrix = np.tile(selection_matrix, (feature_count, 1))
        return selection_matrix

    # Select by quantile
    if quantile is not None:
        def inner_average(x):
            return np.average(x)

        v_inner_average = np.vectorize(inner_average)
        feature_y_average = v_inner_average(feature_partial[:, :, 1])
        quantile_value = np.quantile(feature_y_average, quantile, axis=1, interpolation='nearest').transpose()
        # TODO: Bug fix. When have the same value, multiple value will be seleted.
        selection_matrix = np.isin(feature_y_average, quantile_value)
        # To get indices, one can use where/nonzero or argwhere after isin.

        return selection_matrix


def partial_plot_2d(
        feature_partial, temporal_vector, cluster_vector=None,
        sample_size=None, quantile=None, folder_=folder
):
    feature_count = len(feature_partial)

    # Tile
    temporal_vector = np.tile(temporal_vector.reshape(1, -1), (feature_count, 1))
    if cluster_vector is not None:
        cluster_vector = np.tile(cluster_vector.reshape(1, -1), (feature_count, 1))

    # Do selection
    if sample_size is not None or quantile is not None:
        selection_matrix = select_partial(feature_partial, sample_size, quantile)
        # Order matters
        feature_partial = feature_partial[selection_matrix].reshape(feature_count, -1, 2)
        temporal_vector = temporal_vector[selection_matrix].reshape(feature_count, -1)
        if cluster_vector is not None:
            cluster_vector = cluster_vector[selection_matrix].reshape(feature_count, -1)

    local_count = len(feature_partial[0])
    color_vector = vector_to_color(temporal_vector, stringify=False)

    # Matplotlib Plot Gird
    col = 3
    row = math.ceil(feature_count / col)
    col_length = 3
    row_length = 3
    fig, axs = plt.subplots(
        ncols=col, nrows=row, sharey=True,
        figsize=(col * col_length, row * row_length)
    )
    axs = axs.flatten()

    # Remove null axis
    for ax_remove_index in range(col * row - feature_count):
        fig.delaxes(axs[- ax_remove_index - 1])

    # Iterate each feature
    for feature_index in range(feature_count):
        ax = axs[feature_index]

        for local_index in range(local_count):
            # Matplotlib 2D plot
            ax.plot(
                *feature_partial[feature_index, local_index],
                **{
                    # Receive color tuple/list/array
                    "color": color_vector[feature_index, local_index],
                    "alpha": 0.5, "linewidth": 0.5
                })
            ax.set_title(f'Feature {feature_index}')
            ax.set_xlabel('X Value')
            ax.set_ylabel('Partial Dependent Value')

    plt.tight_layout(h_pad=1.5)
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f'Geo-weighted PDP \n\nSample Size: {sample_size}, Quantile: {quantile}')

    if sample_size is not None:
        suffix = '_Sample' + sample_size
    elif quantile is not None:
        suffix = '_Q' + ';'.join(map(str, quantile))
    else:
        suffix = ''

    plt.savefig(folder_ / f'GeoPDP{suffix}.png')
    plt.close()


def partial_plot_3d(
        feature_partial, temporal_vector, cluster_vector=None,
        sample_size=None, quantile=None, is_ICE=False, labels=None, folder_=folder
):
    """

    Args:
        feature_partial ():
        temporal_vector ():
        cluster_vector (): Shape(N,) or Shape(Feature, N)
        sample_size ():
        quantile ():
        is_ICE ():
        labels ():
        folder_ ():

    Returns:

    """

    feature_count = len(feature_partial)

    # Tile
    temporal_vector = np.tile(temporal_vector.reshape(1, -1), (feature_count, 1))
    if cluster_vector is not None:
        # If Shape(N,). Else Shape(Feature, N).
        if len(cluster_vector.shape) == 1:
            cluster_vector = np.tile(cluster_vector.reshape(1, -1), (feature_count, 1))

    # Do selection
    if sample_size is not None or quantile is not None:
        selection_matrix = select_partial(feature_partial, sample_size, quantile)
        # Order matters
        feature_partial = feature_partial[selection_matrix].reshape(feature_count, -1, 2)
        temporal_vector = temporal_vector[selection_matrix].reshape(feature_count, -1)
        if cluster_vector is not None:
            cluster_vector = cluster_vector[selection_matrix].reshape(feature_count, -1)

    local_count = len(feature_partial[0])

    # Trace name and show_legend control
    if cluster_vector is not None:
        color_vector = vector_to_color(cluster_vector)

        def inner_naming(cluster_label):
            return f'Cluster {cluster_label}'

        v_naming = np.vectorize(inner_naming)
        name_vector = v_naming(cluster_vector)

        def inner_unique(cluster_label):
            _, first_index = np.unique(cluster_label, return_index=True)
            first_vector = np.zeros_like(cluster_label, dtype=bool)
            first_vector[first_index] = True
            return first_vector

        show_vector = np.apply_along_axis(inner_unique, -1, cluster_vector)

    else:
        color_vector = vector_to_color(temporal_vector)
        name_vector = np.empty_like(temporal_vector, dtype=object)
        show_vector = np.zeros_like(temporal_vector, dtype=bool)

    # Iterate each feature
    fig_list = []
    for feature_index in range(feature_count):

        # Each local corresponds to each trace
        trace_list = []
        for local_index in range(local_count):
            x = feature_partial[feature_index, local_index, 0]
            y = feature_partial[feature_index, local_index, 1]
            trace = go.Scatter3d(
                y=x, z=y,
                x=np.tile(temporal_vector[feature_index, local_index], len(x)),
                text=y,
                mode='lines',
                line=dict(
                    # Receive Color String
                    color=color_vector[feature_index, local_index],
                    width=2,
                ),
                name=name_vector[feature_index, local_index],
                legendgroup=name_vector[feature_index, local_index],
                showlegend=bool(show_vector[feature_index, local_index]),
                hovertemplate=
                '<b>X Value</b>: %{y} <br />' +
                '<b>Time Slice</b>: %{x}  <br />' +
                '<b>Partial Value</b>: %{z}  <br />' +
                f'<b>Index</b>: {local_index}  <br />',
            )
            trace_list.append(trace)

        fig = go.Figure(data=trace_list)
        fig.update_layout(
            title={
                'text': f"Temporal Partial Dependency of Feature {feature_index + 1} {labels[feature_index] if labels is not None else ''}",
                'xanchor': 'center',
                'x': 0.45,
                'yanchor': 'top',
                'y': 0.99,
            },
            margin=dict(l=0, r=0, t=50, b=0, pad=0),
            scene=dict(
                xaxis_title='Time Slice',
                yaxis_title='Independent / X value',
                zaxis_title='Dependent / Partial Value',
            ),
            legend_title="Cluster Legend",
            font=dict(
                size=12,
            )
        )

        if sample_size is not None:
            suffix = f'_Sample{sample_size}'
        elif quantile is not None:
            suffix = '_Q' + ';'.join(map(str, quantile))
        else:
            suffix = ''

        fig.write_html(folder_ / f'Geo{"PDP" if not is_ICE else "ICE"}_{feature_index + 1}{suffix}.html')
        fig_list.append(fig)

    return fig_list


def partial_distance(feature_partial):
    """

    Args:
        feature_partial (np.ndarray): Shape(Feature, N, 2)

    Returns:

    """

    feature_count = feature_partial.shape[0]

    # Shape(Feature, N, N)
    feature_distance = []
    # Single feature based cluster. Iterate each feature
    for feature_index in range(feature_count):
        line_distance_matrix = []

        # Iterate each origin data point
        for x_origin, y_origin in feature_partial[feature_index]:
            line_distance_list = []

            # Iterate each dest data point
            for x_dest, y_dest in feature_partial[feature_index]:

                # Overlapped range of two lines. (Max of line start point, Min of line end point)
                overlap_start = max(x_origin[0], x_dest[0])
                overlap_end = min(x_origin[-1], x_dest[-1])

                # No overlapped range.
                if overlap_start >= overlap_end:
                    distance = np.inf
                else:
                    # Get the point in both lines between the overlapped range.
                    x_merge = np.unique(np.concatenate([x_origin, x_dest]))
                    x_merge = x_merge[(overlap_start <= x_merge) & (x_merge <= overlap_end)]

                    # Linear interpolate for the overlapped range.
                    y_merge_origin = np.interp(x_merge, x_origin, y_origin)
                    y_merge_dest = np.interp(x_merge, x_dest, y_dest)

                    # Minimal square distance of two line. Optimal at -b/2a. a is coef of x^2, and b is coef of x.
                    intercept = - np.sum(y_merge_origin - y_merge_dest) / len(x_merge)
                    pointwise_distance = (y_merge_origin - y_merge_dest + intercept) ** 2
                    distance = np.average(pointwise_distance)

                line_distance_list.append(distance)
            line_distance_matrix.append(line_distance_list)

        # Fill Infinity value by max distance.
        line_distance_matrix = np.array(line_distance_matrix)
        line_distance_matrix = np.nan_to_num(line_distance_matrix,
                                             posinf=line_distance_matrix[np.isfinite(line_distance_matrix)].max() * 2)

        feature_distance.append(line_distance_matrix)

    return np.array(feature_distance)


def partial_cluster(feature_partial, n_clusters=4):
    """
    Cluster data point based on partial dependency

    Args:
        feature_partial (np.ndarray): Shape(Feature, N, 2)
        n_clusters ():

    Returns:
        feature_distance, feature_cluster_label, distance_matrix, cluster_label

    """

    feature_count = feature_partial.shape[0]

    feature_distance = partial_distance(feature_partial)

    # Shape(Feature, N)
    feature_cluster_label = []
    for feature_index in range(feature_count):
        # Fit cluster model again for N cluster label (Or distance threshold)
        # TODO: No affinity set as precomputed in the order version.
        model = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='complete')
        model.fit(feature_distance[feature_index])

        # Record feature label result
        feature_cluster_label.append(model.labels_)
    feature_cluster_label = np.array(feature_cluster_label)

    # Multi-feature based cluster. Shape(N, Feature)
    X = feature_cluster_label.T

    # Calculate the distance based on features clustering result. Shape(N, N)
    distance_matrix = []
    for origin in X:
        distance_list = []
        for dest in X:
            dis = np.count_nonzero(origin != dest)
            distance_list.append(dis)
        distance_matrix.append(distance_list)

    model = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='complete')
    model.fit(distance_matrix)
    cluster_label = model.labels_

    return feature_distance, feature_cluster_label, distance_matrix, cluster_label


def cluster_dendrogram_plot(distance_matrix, cluster_vector=None):
    # Fit cluster model for hierarchy plot
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='precomputed', linkage='complete')
    model.fit(distance_matrix)

    # Plot the hierarchy figure
    plt.figure(figsize=(8, 6))
    # TODO: Parameter decision
    linkage_matrix = plot_dendrogram(model, truncate_mode="level", p=3)

    # TODO: Fix label overflow.
    # plt.tight_layout()

    # Save and clear the plot context
    plt.ylabel('Variance of the clusters')
    plt.xlabel("Number of points in node (or index of point if no parenthesis)")

    return plt


def partial_cluster_plot(
        feature_distance, feature_cluster_label_list, distance_matrix, cluster_label,
        geo_vector, temporal_vector, cluster_vector=None, labels=None, folder_=folder
):
    feature_count = feature_distance.shape[0]

    # Single feature based cluster. Iterate each feature
    for feature_index in range(feature_count):
        cluster_dendrogram_plot(feature_distance[feature_index])
        plt.title(
            f'Hierarchy Plot of Feature {feature_index + 1} {labels[feature_index] if labels is not None else ""}')
        # TODO: Save to where?
        plt.savefig(folder_ / f'Hierarchy_{feature_index + 1}.png')
        plt.clf()

        # Plot the single feature cluster result
        scatter_3d(
            geo_vector, temporal_vector, feature_cluster_label_list[feature_index],
            f'Spatio-temporal Cluster Plot of Feature {feature_index + 1} {labels[feature_index] if labels is not None else ""}',
            'Cluster Label',
            filename=f'Cluster_{feature_index + 1}', is_cluster=True, folder_=folder_)

    cluster_dendrogram_plot(distance_matrix)
    plt.title('Hierarchy Plot of Integrated Feature')
    plt.savefig(folder_ / f'Hierarchy_Integrated.png')
    plt.clf()

    scatter_3d(
        geo_vector, temporal_vector, cluster_label,
        f'Integrated Spatio-temporal Cluster Plot', 'Cluster Label',
        filename=f'Cluster_Integrated', is_cluster=True, folder_=folder_)


def show_plain_partial(X, local_estimator_list):
    """
    PDP without considering local weight effect
    """

    local_count = X.shape[0]
    feature_count = X.shape[1]

    ax = None
    display = None
    for local_estimator in local_estimator_list:
        display = PartialDependenceDisplay.from_estimator(
            local_estimator,
            X,
            range(feature_count),
            kind="average",
            n_jobs=-1,
            grid_resolution=20,
            random_state=0,
            ice_lines_kw={"color": "tab:blue", "alpha": 0.2, "linewidth": 0.5},
            pd_line_kw={"color": "tab:orange", "linestyle": "--"},
            ax=ax
        )
        ax = display.axes_

    figure = display.figure_
    figure.set_figwidth(10)
    figure.set_figheight(30)
    figure.tight_layout()
    figure.suptitle("Partial dependence without local weight")
    figure.subplots_adjust(hspace=0.3)
    plt.savefig(folder / 'PDP.png')


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

    return linkage_matrix
