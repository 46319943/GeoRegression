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
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.inspection import PartialDependenceDisplay
from hdbscan import HDBSCAN
from umap import UMAP
import plotly.graph_objects as go
import plotly.express as px

from scipy.stats import logistic

from georegression.visualize.scatter import scatter_3d
from georegression.visualize.utils import vector_to_color, range_margin

from georegression.visualize import default_folder


def select_partial(feature_partial, sample_size=None, quantile=None):
    """

    Args:
        feature_partial (np.ndarray): Shape(Feature, N, 2)
        sample_size (): Int for specific count. Float for rate.
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
        feature_partial, cluster_vector, cluster_typical,
        weight_style=True, alpha_range=None, width_range=None, use_sigmoid=True, scale_power=1,
        folder_=default_folder
):
    """

    Args:

        feature_partial (): Shape(Feature, N, 2)
        cluster_vector (): Shape(N,) or Shape(Feature, N)
        cluster_typical (): Shape(n_cluster) or Shape(Feature, n_cluster)
        alpha_range ():
        width_range ():
        scale_power ():
        use_sigmoid ():
        weight_style (bool):
        folder_ ():

    Returns:

    """

    if alpha_range is None:
        alpha_range = [0.1, 1]
    if width_range is None:
        width_range = [0.5, 3]

    if len(cluster_vector.shape) == 1:
        is_integrated = True
    else:
        is_integrated = False

    feature_count = len(feature_partial)
    local_count = len(feature_partial[0])

    # Matplotlib Plot Gird
    col = 3
    row = math.ceil(feature_count / col)
    col_length = 3
    row_length = 2

    # Close interactive mode
    plt.ioff()

    fig, axs = plt.subplots(
        ncols=col, nrows=row, sharey='none',
        figsize=(col * col_length, (row + 1) * row_length)
    )

    # Set figure size after creating to avoid screen resize.
    if plt.isinteractive():
        plt.gcf().set_size_inches(col * col_length, (row + 1) * row_length)

    # 2d-ndarray flatten
    axs = axs.flatten()

    # Remove null axis
    for ax_remove_index in range(col * row - feature_count):
        fig.delaxes(axs[- ax_remove_index - 1])

    # Iterate each feature
    for feature_index in range(feature_count):
        ax = axs[feature_index]

        if is_integrated:
            inner_vector = np.copy(cluster_vector)
            inner_typical = np.copy(cluster_typical)
        else:
            inner_vector = cluster_vector[feature_index]
            inner_typical = cluster_typical[feature_index]

        # Style the line by the cluster size.
        values, counts = np.unique(inner_vector, return_counts=True)
        if np.max(counts) == np.min(counts):
            style_ratios = np.ones(local_count)
        else:
            # style_ratios = (counts - np.min(counts)) / (np.max(counts) - np.min(counts))
            style_ratios = counts / local_count
            if use_sigmoid:
                style_ratios = (style_ratios - 0.5) * 10
                style_ratios = logistic.cdf(style_ratios)
            style_ratios = style_ratios ** scale_power
        # np.xx_like returns array having the same type as input array.
        style_alpha = np.zeros(local_count)
        style_width = np.zeros(local_count)
        for value, style_ratio in zip(values, style_ratios):
            cluster_index = np.nonzero(inner_vector == value)
            style_alpha[cluster_index] = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * style_ratio
            style_width[cluster_index] = width_range[0] + (width_range[1] - width_range[0]) * style_ratio

        # Cluster typical selection
        inner_partial = feature_partial[feature_index, inner_typical]
        inner_vector = inner_vector[inner_typical]
        style_alpha = style_alpha[inner_typical]
        style_width = style_width[inner_typical]

        color_vector = vector_to_color(inner_vector, stringify=False)

        for local_index in range(len(inner_partial)):
            # Matplotlib 2D plot
            ax.plot(
                *inner_partial[local_index],
                **{
                    # Receive color tuple/list/array
                    "color": color_vector[local_index],
                    "alpha": style_alpha[local_index], "linewidth": style_width[local_index],
                    "label": f'Cluster {inner_vector[local_index]}'
                })
            ax.set_title(f'Feature {feature_index + 1}')

        # Individual file for each feature
        fig_ind = plt.figure(figsize=(5, 4), constrained_layout=True)
        for local_index in range(len(inner_partial)):
            fig_ind.gca().plot(
                *inner_partial[local_index],
                **{
                    # Receive color tuple/list/array
                    "color": color_vector[local_index],
                    "alpha": style_alpha[local_index], "linewidth": style_width[local_index],
                    "label": f'Cluster {inner_vector[local_index]}'
                }
            )
        plt.xlabel('Independent Value')
        plt.ylabel('Partial Dependent Value')
        plt.title(f'SPPDP of Typical Cluster in Feature {feature_index + 1}', pad=60)
        plt.legend(
            loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=5,
            columnspacing=0.2, fontsize='x-small', numpoints=2
        )
        fig_ind.savefig(
            folder_ / f'SPPDP_Typical{"_Merged" if is_integrated else ""}{feature_index + 1}',
            dpi=300
        )
        fig_ind.clear()

    fig.supxlabel('Independent Value')
    fig.supylabel('Partial Dependent Value')

    fig.tight_layout(h_pad=1.5)
    fig.subplots_adjust(top=0.85)
    fig.suptitle(f'SPPDP')

    if is_integrated:
        handles, labels = ax.get_legend_handles_labels()
        # put the center upper edge of the bounding box at the coordinates(bbox_to_anchor)
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.965), ncol=6)

    fig.savefig(folder_ / f'SPPDP{"_Merged" if is_integrated else ""}.png')
    fig.clear()


def partial_plot_3d(
        feature_partial, temporal_vector, cluster_vector=None,
        sample_size=None, quantile=None, is_ICE=False, labels=None, folder=default_folder
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
        folder ():

    Returns:

    """

    feature_count = len(feature_partial)

    # Tile
    temporal_vector = np.tile(temporal_vector.reshape(1, -1), (feature_count, 1))
    index_label = np.tile(np.arange(len(feature_partial[0])).reshape(1, -1), (feature_count, 1))

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
        index_label = index_label[selection_matrix].reshape(feature_count, -1)
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
                f'<b>Index</b>: {index_label[feature_index, local_index]}  <br />' +
                '<b>Partial Value</b>: %{z}  <br />'

            )
            trace_list.append(trace)

        fig = go.Figure(data=trace_list)
        fig.update_layout(
            title={
                'text': f"SPPDP of Feature {feature_index + 1} {labels[feature_index] if labels is not None else ''}",
                'xanchor': 'center',
                'x': 0.45,
                'yanchor': 'top',
                'y': 0.99,
            },
            margin=dict(l=0, r=0, t=50, b=0, pad=0),
            legend_title="Cluster Legend",
            font=dict(
                size=12,
            ),
            template="seaborn",
            font_family="Times New Roman"
        )

        # Fix range while toggling trace.
        y_max = np.max([np.max(y) for y in feature_partial[feature_index, :, 1]])
        y_min = np.min([np.min(y) for y in feature_partial[feature_index, :, 1]])

        x_max = np.max([np.max(x) for x in feature_partial[feature_index, :, 0]])
        x_min = np.min([np.min(x) for x in feature_partial[feature_index, :, 0]])

        fig.update_scenes(
            xaxis_title='Time Slice',
            xaxis_range=range_margin(vector=temporal_vector),
            yaxis_title='Independent / X value',
            yaxis_range=range_margin(value_min=x_min, value_max=x_max),
            zaxis_title='Dependent / Partial Value',
            zaxis_range=range_margin(value_min=y_min, value_max=y_max),
        )

        if sample_size is not None:
            suffix = f'_Sample{sample_size}'
        elif quantile is not None:
            suffix = '_Q' + ';'.join(map(str, quantile))
        else:
            suffix = ''

        fig.write_html(
            folder / f'SP{"PDP" if not is_ICE else "ICE"}{"_Merged" if is_integrated else ""}_{feature_index + 1}{suffix}.html')
        fig_list.append(fig)

    return fig_list


def partial_distance(partial):
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

                # Weighting according to the point interval in the bi-direction.
                distance_weight = np.zeros_like(x_merge)
                distance_weight[1:-1] = x_merge[2:] - x_merge[:-2]
                distance_weight[0] = (x_merge[1] - x_merge[0]) * 2
                distance_weight[-1] = (x_merge[-1] - x_merge[-2]) * 2
                distance_weight = distance_weight / np.sum(distance_weight)

                distance = np.average(pointwise_distance, weights=distance_weight)

            line_distance_list.append(distance)
        line_distance_matrix[origin_index, origin_index:] = line_distance_list

    # Fill Infinity value by max distance.
    line_distance_matrix = np.nan_to_num(line_distance_matrix,
                                         posinf=line_distance_matrix[np.isfinite(line_distance_matrix)].max() * 2)

    # Fill the up triangular matrix.
    line_distance_matrix = line_distance_matrix + np.transpose(line_distance_matrix)

    return line_distance_matrix


def features_partial_distance(features_partial):
    """
    Calculation distance between partial lines.

    Args:
        features_partial (np.ndarray): Shape(Feature, N, 2)

    Returns:
        feature_distance (np.ndarray): Shape(Feature, N, N)

    """

    feature_count = features_partial.shape[0]

    # Shape(Feature, N, N)
    features_distance = Parallel(n_jobs=-1)(
        # Single feature based cluster. Iterate each feature
        delayed(partial_distance)(features_partial[feature_index])
        for feature_index in range(feature_count)
    )

    return np.array(features_distance)


def partial_cluster(
        partial=None, distance=None,
        n_neighbours=5, min_dist=0.1, n_components=2,
        min_cluster_size=10, min_samples=3, cluster_selection_epsilon=1,

        select_clusters=False,
        plot_title=None, plot_filename=None, plot_folder=default_folder,
):
    """
    Cluster data based on partial dependence result or derived distance matrix.

    Args:

        partial (np.ndarray): Shape(N, 2)
        distance (np.ndarray): Shape(N, N)
        n_neighbours:
        min_dist:
        n_components:
        min_cluster_size:
        min_samples:
        cluster_selection_epsilon:
        select_clusters:
        plot_filename:
        plot_title:
        plot_folder:

    Returns:

    """

    if plot_title is None:
        plot_title = f'Condensed trees'
    if plot_filename is None:
        plot_filename = f'CondensedTrees.png'

    # Parameter check
    if partial is None and distance is None:
        raise Exception('Feature partial or feature distance matrix should be provided.')

    # Ensure feature distance is available.
    if distance is None:
        distance = partial_distance(partial)

    # TODO: Range of UMAP embedding value?
    # Reduce dimension. Mapping the distance matrix to low dimension space embedding.
    # Standard embedding is used for visualization. Clusterable embedding is used for clustering.
    standard_embedding = UMAP(
        random_state=42, n_neighbors=n_neighbours, min_dist=min_dist, metric='precomputed'
    ).fit_transform(distance)
    if n_components == 2:
        clusterable_embedding = standard_embedding
    else:
        clusterable_embedding = UMAP(
            random_state=42, n_neighbors=n_neighbours, min_dist=min_dist, n_components=n_components,
            metric='precomputed'
        ).fit_transform(distance)

    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                    cluster_selection_epsilon=cluster_selection_epsilon
                    ).fit(clusterable_embedding)

    model.condensed_tree_.plot(select_clusters=select_clusters)
    plt.title(plot_title)
    plt.savefig(plot_folder / plot_filename)
    plt.clf()

    return standard_embedding, model.labels_, distance


def features_partial_cluster(
        features_partial=None, features_distance=None,
        n_neighbours=5, min_dist=0.1, n_components=2,
        min_cluster_size=10, min_samples=3, cluster_selection_epsilon=1,
        select_clusters=False,
        labels=None, only_integrated=False, folder=default_folder,
):
    """
    Cluster data point based on partial dependency

    Args:
        labels (): Feature labels.
        features_distance ():
        folder ():
        n_neighbours ():
        min_dist ():
        n_components ():
        min_cluster_size ():
        min_samples ():
        cluster_selection_epsilon ():
        features_partial (np.ndarray): Shape(Feature, N, 2)
        select_clusters ():

    Returns:
        feature_embedding, feature_cluster_label, cluster_embedding, cluster_label

    """

    # TODO: More fine-tuning control on the multi-features and integrate-feature.

    # Parameter check
    if features_partial is None and features_distance is None:
        raise Exception('Feature partial or feature distance should be provided.')

    # Ensure feature distance is available.
    if features_distance is None:
        features_distance = features_partial_distance(features_partial)

    # Integrated feature cluster

    # Sum the distance over each feature.
    # Shape(Feature, N, N) -> Shape(N, N).
    distance = np.sum(features_distance, axis=0)
    cluster_embedding, cluster_label, _ = partial_cluster(distance=distance, n_neighbours=n_neighbours * 2, min_dist=min_dist * 2,
                                                          n_components=n_components, min_cluster_size=min_cluster_size, min_samples=min_samples,
                                                          cluster_selection_epsilon=cluster_selection_epsilon,
                                                          select_clusters=select_clusters,
                                                          plot_title=f'Condensed trees of total features',
                                                          plot_filename=f'CondensedTrees_Integrated.png',
                                                          plot_folder=folder
                                                          )

    # Return the result if only the integrated result is required.
    if only_integrated:
        return cluster_embedding, cluster_label

    # Individual feature cluster
    feature_count = features_distance.shape[0]

    # Shape(Feature, N, 2)
    features_embedding = []
    # Shape(Feature, N)
    features_cluster_label = []
    for feature_index in range(feature_count):
        cluster_embedding, cluster_label, _ = partial_cluster(
            partial=features_partial[feature_index] if features_partial else None, distance=features_distance[feature_index],
            n_neighbours=n_neighbours, min_dist=min_dist, n_components=n_components,
            min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon,
            select_clusters=select_clusters,
            plot_title=f'Condensed trees of Feature {feature_index + 1} {labels[feature_index] if labels is not None else ""}',
            plot_filename=f'CondensedTrees_{feature_index + 1}.png',
            plot_folder=folder

        )

        # Record feature label result
        features_cluster_label.append(cluster_label)
        features_embedding.append(cluster_embedding)

    features_cluster_label = np.array(features_cluster_label)
    features_embedding = np.array(features_embedding)

    return features_embedding, features_cluster_label, cluster_embedding, cluster_label


def choose_cluster_typical(embedding, cluster_vector):
    """
    Return the index of typical items for each cluster.
    The typical item of a cluster is the centre of the cluster,
    which has the minimal summation of distance to others in the same cluster.

    Args:
        embedding ():
        cluster_vector ():

    Returns:

    """
    cluster_typical_list = []
    cluster_value = np.unique(cluster_vector)
    for cluster in cluster_value:
        cluster_index_vector = np.nonzero(cluster_vector == cluster)[0]
        embedding_cluster = embedding[cluster_index_vector]
        cluster_typical_list.append(
            cluster_index_vector[np.argmin(np.sum(squareform(pdist(embedding_cluster)), axis=1))]
        )

    return cluster_typical_list


def embedding_plot(
        embedding, cluster, temporal_vector,
        title, filename, folder_=default_folder
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
        template="seaborn",
        font_family="Times New Roman"
    )

    fig.update_xaxes(
        title="Embedding dimension X",
        range=range_margin(embedding[:, 0])
    )
    fig.update_yaxes(
        title="Embedding dimension Y",
        range=range_margin(embedding[:, 1]),
        scaleanchor="x",
        scaleratio=1,
    )

    fig.write_html(folder_ / f'{filename}.html')

    return fig


def compass_plot(
        cluster_fig, partial_fig, embedding_fig,
        plot_filename="SPPDP_Compass.html", plot_folder=default_folder
):
    """
    Subplots of 2 rows and 2 columns.
    [cluster plot, partial plot  ]
    [cluster plot, embedding plot]

    """

    fig = make_subplots(
        cols=2, rows=2,
        column_widths=[0.5, 0.5], row_heights=[0.6, 0.4],
        horizontal_spacing=0.02, vertical_spacing=0.05,
        specs=[
            [{'rowspan': 2, "type": "scene"}, {"type": "scene"}],
            [None, {"type": "xy"}]
        ],
        subplot_titles=(
            cluster_fig.layout.title.text,
            partial_fig.layout.title.text,
            embedding_fig.layout.title.text)
    )

    fig.add_traces(cluster_fig.data, rows=1, cols=1)
    fig.add_traces(partial_fig.data, rows=1, cols=2)
    fig.add_traces(embedding_fig.data, rows=2, cols=2)

    fig.update_layout(cluster_fig.layout)
    fig.update_scenes(cluster_fig.layout.scene, row=1, col=1)
    fig.update_scenes(partial_fig.layout.scene, row=1, col=2)
    fig.update_xaxes(embedding_fig.layout.xaxis, row=2, col=2)
    fig.update_yaxes(embedding_fig.layout.yaxis, row=2, col=2)
    fig.update_layout(title_text='SPPDP Compass')

    fig.write_html(plot_folder / plot_filename)

    return fig


def partial_compound_plot(
        geo_vector, temporal_vector, feature_partial,
        feature_embedding, feature_cluster_label,
        cluster_embedding, cluster_label,
        sample_size=None,
        labels=None, folder=default_folder,
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
        sample_size ():
        labels (): Shape(Feature)
        folder ():


    Returns:

    """

    # TODO: Add hover highlight.

    feature_count = len(feature_partial)

    # Each feature clusters.
    partial_fig_list = partial_plot_3d(
        feature_partial, temporal_vector, feature_cluster_label,
        sample_size=sample_size, labels=labels, folder=folder
    )

    # Whole features cluster.
    partial_plot_integrated_list = partial_plot_3d(
        feature_partial, temporal_vector, cluster_label,
        sample_size=sample_size, labels=labels, folder=folder
    )
    embedding_integrated_plot = embedding_plot(
        cluster_embedding, cluster_label, temporal_vector,
        f'Low dimension embedding of total features',
        f'UMAP_Merged', folder_=folder
    )
    cluster_integrated_plot = scatter_3d(
        geo_vector, temporal_vector, cluster_label,
        f'Merged Spatio-temporal Cluster Plot', 'Cluster Label',
        filename=f'Cluster_Merged', is_cluster=True, folder=folder)

    cluster_fig_list = []
    embedding_fig_list = []
    for feature_index in range(feature_count):
        embedding_fig = embedding_plot(
            feature_embedding[feature_index], feature_cluster_label[feature_index], temporal_vector,
            f'Low dimension embedding of Feature {feature_index + 1} {labels[feature_index] if labels is not None else ""}',
            f'UMAP_{feature_index + 1}', folder_=folder
        )
        embedding_fig_list.append(embedding_fig)

        # Plot the single feature cluster result
        cluster_plot = scatter_3d(
            geo_vector, temporal_vector, feature_cluster_label[feature_index],
            f'Spatio-temporal Cluster Plot of Feature {feature_index + 1} {labels[feature_index] if labels is not None else ""}',
            'Cluster Label',
            filename=f'Cluster_{feature_index + 1}', is_cluster=True, folder=folder)
        cluster_fig_list.append(cluster_plot)

        compass_plot(
            cluster_plot, partial_fig_list[feature_index], embedding_fig,
            plot_filename=f'SPPDP_Compass_{feature_index + 1}.html', plot_folder=folder
        )

        compass_plot(
            cluster_integrated_plot, partial_plot_integrated_list[feature_index], embedding_integrated_plot,
            plot_filename=f'SPPDP_Merged_Compass_{feature_index + 1}.html', plot_folder=folder
        )

    return cluster_fig_list, partial_fig_list, embedding_fig_list
