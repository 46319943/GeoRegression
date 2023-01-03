import math
import time
from os.path import join
from pathlib import Path

import matplotlib
import numpy as np
from joblib import Parallel, delayed
from matplotlib import cm, pyplot as plt
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.inspection import PartialDependenceDisplay
from hdbscan import HDBSCAN
from umap import UMAP
import plotly.graph_objects as go
import plotly.express as px

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
        # TODO: Bug fix. When have the same value, multiple value will be selected.
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
        ncols=col, nrows=row, sharey='all',
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

    # Feature cluster or Integrated cluster.
    is_integrated = False
    if cluster_vector is not None:
        # If Shape(N,). Else Shape(Feature, N).
        if len(cluster_vector.shape) == 1:
            is_integrated = True
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
                f'<b>Index</b>: {local_index}  <br />' +
                '<b>Partial Value</b>: %{z}  <br />'

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
            legend_title="Cluster Legend",
            font=dict(
                size=12,
            )
        )
        fig.update_scenes(
            xaxis_title='Time Slice',
            yaxis_title='Independent / X value',
            zaxis_title='Dependent / Partial Value',
        )

        if sample_size is not None:
            suffix = f'_Sample{sample_size}'
        elif quantile is not None:
            suffix = '_Q' + ';'.join(map(str, quantile))
        else:
            suffix = ''

        fig.write_html(
            folder_ / f'Geo{"PDP" if not is_ICE else "ICE"}{"_Integrated" if is_integrated else ""}_{feature_index + 1}{suffix}.html')
        fig_list.append(fig)

    return fig_list


def feature_partial_distance(partial):
    """
    Calculation distance between partial lines.

    Args:
        partial (np.ndarray): partial result of a feature. Shape(N, 2)

    Returns:
        distance_matrix (np.ndarray): Shape(N, N)
    """

    N = partial.shape[0]
    line_distance_matrix = np.zeros((N, N))

    # Iterate each origin data point
    for origin_index, (x_origin, y_origin) in enumerate(partial):
        line_distance_list = []

        # Iterate each dest data point
        for x_dest, y_dest in partial[origin_index:]:

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
        line_distance_matrix[origin_index, origin_index:] = line_distance_list

    # Fill Infinity value by max distance.
    line_distance_matrix = np.nan_to_num(line_distance_matrix,
                                         posinf=line_distance_matrix[np.isfinite(line_distance_matrix)].max() * 2)

    # Fill the up triangular matrix.
    line_distance_matrix = line_distance_matrix + np.transpose(line_distance_matrix)

    return line_distance_matrix


def partial_distance(feature_partial):
    """
    Calculation distance between partial lines.

    Args:
        feature_partial (np.ndarray): Shape(Feature, N, 2)

    Returns:
        feature_distance (np.ndarray): Shape(Feature, N, N)

    """

    feature_count = feature_partial.shape[0]

    # Shape(Feature, N, N)
    feature_distance = Parallel(n_jobs=-1)(
        # Single feature based cluster. Iterate each feature
        delayed(feature_partial_distance)(feature_partial[feature_index])
        for feature_index in range(feature_count)
    )

    return np.array(feature_distance)


def partial_cluster(
        geo_vector, temporal_vector,
        feature_partial=None, feature_distance=None,
        n_neighbours=5, min_dist=0.1, n_components=2,
        min_cluster_size=10, min_samples=3, cluster_selection_epsilon=1,
        select_clusters=False,
        labels=None, folder_=folder,
):
    """
    Cluster data point based on partial dependency

    Args:
        geo_vector ():
        temporal_vector ():
        labels (): Feature labels.
        feature_distance ():
        folder_ ():
        n_neighbours ():
        min_dist ():
        n_components ():
        min_cluster_size ():
        min_samples ():
        cluster_selection_epsilon ():
        feature_partial (np.ndarray): Shape(Feature, N, 2)
        select_clusters ():

    Returns:
        feature_distance, feature_cluster_label, distance_matrix, cluster_label

    """

    if feature_partial is None and feature_distance is None:
        raise Exception('Feature partial or feature distance should be provided.')

    if feature_distance is None:
        feature_distance = partial_distance(feature_partial)

    feature_count = feature_distance.shape[0]

    # Shape(Feature, N, 2)
    feature_embedding = []
    # Shape(Feature, N)
    feature_cluster_label = []
    for feature_index in range(feature_count):
        # TODO: Range of UMAP embedding value?
        # Dimension reduction. Standard for visualization. Clusterable for clustering.
        standard_embedding = UMAP(
            random_state=42, n_neighbors=n_neighbours, min_dist=min_dist, metric='precomputed'
        ).fit_transform(feature_distance[feature_index])
        if n_components == 2:
            clusterable_embedding = standard_embedding
        else:
            clusterable_embedding = UMAP(
                random_state=42, n_neighbors=n_neighbours, min_dist=min_dist, n_components=n_components,
                metric='precomputed'
            ).fit_transform(feature_distance[feature_index])

        model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                        cluster_selection_epsilon=cluster_selection_epsilon
                        ).fit(clusterable_embedding)

        # Record feature label result
        feature_cluster_label.append(model.labels_)
        feature_embedding.append(standard_embedding)

        # Plot cluster tree
        model.condensed_tree_.plot(select_clusters=select_clusters)
        plt.title(
            f'Condensed trees of Feature {feature_index + 1} {labels[feature_index] if labels is not None else ""}'
        )
        plt.savefig(folder_ / f'CondensedTrees_{feature_index + 1}.png')
        plt.clf()

    feature_cluster_label = np.array(feature_cluster_label)
    feature_embedding = np.array(feature_embedding)

    # Distance based on feature distance.
    distance_matrix = np.sum(feature_distance, axis=0)

    standard_embedding = UMAP(
        random_state=42, n_neighbors=n_neighbours * 2, min_dist=min_dist * 2, metric='precomputed'
    ).fit_transform(distance_matrix)

    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                    cluster_selection_epsilon=cluster_selection_epsilon
                    ).fit(standard_embedding)

    cluster_label = model.labels_
    cluster_embedding = standard_embedding

    model.condensed_tree_.plot(select_clusters=select_clusters)
    plt.title(
        f'Condensed trees of total features'
    )
    plt.savefig(folder_ / f'CondensedTrees_Integrated.png')
    plt.clf()

    return feature_embedding, feature_cluster_label, cluster_embedding, cluster_label


def embedding_plot(
        embedding, cluster, temporal_vector,
        title, filename, folder_=folder
):
    """
    2D Embedding plot colored by cluster.

    Args:
        embedding (np.ndarray): Shape(N, 2)
        cluster (): Shape(N,)
        temporal_vector (): Shape(N,)
        title ():
        filename ():
        folder_ ():

    Returns:

    """
    fig = go.Figure()

    local_index = np.arange(embedding.shape[0]).reshape(-1, 1)
    custom_data = np.concatenate([temporal_vector, local_index], axis=1)

    color = vector_to_color(cluster)

    for cluster_value in np.unique(cluster):
        cluster_index = cluster == cluster_value
        fig.add_trace(
            go.Scattergl(
                x=embedding[cluster_index, 0], y=embedding[cluster_index, 1],
                customdata=custom_data[cluster_index], mode='markers',
                # Name of trace for legend display
                name=f'Cluster {cluster_value}',
                legendgroup=f'Cluster {cluster_value}',
                marker={
                    'color': color[cluster_index],
                    'size': 5,
                },
                text=cluster[cluster_index],
                hovertemplate=
                f'<b>Cluster</b> :' + ' %{text} <br />' +
                f'<b>Time Slice</b> :' + ' %{customdata[0]} <br />' +
                f'<b>Index</b> :' + ' %{customdata[1]} <br />' +
                '<extra></extra>',
            )
        )

    fig.update_layout(
        title=title,
        legend_title="clusters",
    )

    fig.update_xaxes(title="Embedding dimension X")
    fig.update_yaxes(title="Embedding dimension Y")

    fig.write_html(folder_ / f'{filename}.html')

    return fig


def partial_compound_plot(
        geo_vector, temporal_vector, feature_partial,
        feature_embedding, feature_cluster_label,
        cluster_embedding, cluster_label,
        labels=None, folder_=folder,
):
    """
    Subplots of 2 rows and 2 columns.
    [cluster plot, partial plot  ]
    [cluster plot, embedding plot]

    One compound plot for each feature cluster result. Another compound plot for whole feature cluster result.

    Args:
        geo_vector ():
        temporal_vector ():
        feature_partial (): Shape(Feature, N, 2)
        feature_embedding (): Shape(Feature, N, 2)
        feature_cluster_label (): Shape(Feature, N)
        cluster_embedding (): Shape(N, 2)
        cluster_label (): Shape(N,)
        labels (): Shape(Feature)
        folder_ ():


    Returns:

    """

    # TODO: Add hover highlight.

    feature_count = len(feature_partial)

    # Each feature clusters.
    partial_plot_list = partial_plot_3d(
        feature_partial, temporal_vector, feature_cluster_label,
        sample_size=0.2, labels=labels, folder_=folder
    )

    # Whole features cluster.
    partial_plot_integrated_list = partial_plot_3d(
        feature_partial, temporal_vector, cluster_label,
        sample_size=0.2, labels=labels, folder_=folder
    )
    embedding_integrated_plot = embedding_plot(
        cluster_embedding, cluster_label, temporal_vector,
        f'Low dimension embedding of total features',
        f'UMAP_Integrated', folder_=folder_
    )
    cluster_integrated_plot = scatter_3d(
        geo_vector, temporal_vector, cluster_label,
        f'Integrated Spatio-temporal Cluster Plot', 'Cluster Label',
        filename=f'Cluster_Integrated', is_cluster=True, folder_=folder_)

    for feature_index in range(feature_count):
        embedding_fig = embedding_plot(
            feature_embedding[feature_index], feature_cluster_label[feature_index], temporal_vector,
            f'Low dimension embedding of Feature {feature_index + 1} {labels[feature_index] if labels is not None else ""}',
            f'UMAP_{feature_index + 1}', folder_=folder_
        )

        # Plot the single feature cluster result
        cluster_plot = scatter_3d(
            geo_vector, temporal_vector, feature_cluster_label[feature_index],
            f'Spatio-temporal Cluster Plot of Feature {feature_index + 1} {labels[feature_index] if labels is not None else ""}',
            'Cluster Label',
            filename=f'Cluster_{feature_index + 1}', is_cluster=True, folder_=folder_)

        fig = make_subplots(
            cols=2, rows=2,
            column_widths=[0.5, 0.5], row_heights=[0.6, 0.4],
            horizontal_spacing=0.02, vertical_spacing=0.05,
            specs=[
                [{'rowspan': 2, "type": "scene"}, {"type": "scene"}],
                [None, {"type": "xy"}]
            ],
            subplot_titles=(
                cluster_plot.layout.title.text,
                partial_plot_list[feature_index].layout.title.text,
                embedding_fig.layout.title.text)
        )

        fig.add_traces(cluster_plot.data, rows=1, cols=1)
        fig.add_traces(partial_plot_list[feature_index].data, rows=1, cols=2)
        fig.add_traces(embedding_fig.data, rows=2, cols=2)

        fig.update_layout(cluster_plot.layout)
        fig.update_scenes(cluster_plot.layout.scene, row=1, col=1)
        fig.update_scenes(partial_plot_list[feature_index].layout.scene, row=1, col=2)
        fig.update_xaxes(title="Embedding dimension X", row=2, col=2)
        fig.update_yaxes(title="Embedding dimension Y", row=2, col=2)
        fig.update_layout(title_text='Partial Compound Plot')

        fig.write_html(folder_ / f'PartialCompound_{feature_index + 1}.html')

        fig_integrated = make_subplots(
            cols=2, rows=2,
            column_widths=[0.5, 0.5], row_heights=[0.6, 0.4],
            horizontal_spacing=0.02, vertical_spacing=0.05,
            specs=[
                [{'rowspan': 2, "type": "scene"}, {"type": "scene"}],
                [None, {"type": "xy"}]
            ],
            subplot_titles=(
                cluster_integrated_plot.layout.title.text,
                partial_plot_integrated_list[feature_index].layout.title.text,
                embedding_integrated_plot.layout.title.text)
        )

        fig_integrated.add_traces(cluster_integrated_plot.data, rows=1, cols=1)
        fig_integrated.add_traces(partial_plot_integrated_list[feature_index].data, rows=1, cols=2)
        fig_integrated.add_traces(embedding_integrated_plot.data, rows=2, cols=2)

        fig_integrated.update_layout(cluster_integrated_plot.layout)
        fig_integrated.update_scenes(cluster_integrated_plot.layout.scene, row=1, col=1)
        fig_integrated.update_scenes(partial_plot_integrated_list[feature_index].layout.scene, row=1, col=2)
        fig_integrated.update_xaxes(title="Embedding dimension X", row=2, col=2)
        fig_integrated.update_yaxes(title="Embedding dimension Y", row=2, col=2)
        fig_integrated.update_layout(title_text='Partial Compound Plot')

        fig_integrated.write_html(folder_ / f'PartialCompound_Integrated_{feature_index + 1}.html')

    return
