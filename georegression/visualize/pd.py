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

import plotly.graph_objects as go
import plotly.express as px

from scipy.stats import logistic

from georegression.visualize.scatter import scatter_3d
from georegression.visualize.utils import vector_to_color, range_margin

from georegression.visualize import default_folder


def sample_partial(partial, sample_size=None, quantile=None, cluster_label=None, random_state=1003):
    """
    Use random sample or quantile/percentile to get the subset of partial data.

    Args:
        partial (np.ndarray): Shape(N, 2)
        sample_size (): Int for specific count. Float for rate.
        quantile ():
        random_state:

    Returns:

    """
    # Set random state
    if random_state is not None:
        np.random.seed(random_state)

    N = partial.shape[0]

    if sample_size is None and quantile is None:
        raise Exception('No selection method is chosen.')
    if sample_size is not None and quantile is not None:
        raise Exception('Only one selection method is allowed.')

    # Select by sample
    if sample_size is not None:
        # Proportional sample.
        if isinstance(sample_size, float):
            sample_size = int(sample_size * N)

        # Ensure at least one sample for each cluster.
        if cluster_label is not None:
            # Sample size is proportional to cluster size. bincount cannot handle negative values (-1 for un-clustered label).
            cluster_values, cluster_sizes = np.unique(cluster_label, return_counts=True)
            cluster_sample_sizes = np.ceil(cluster_sizes * sample_size / N).astype(int)
            # Ensure at least one sample for each cluster. Sample size is no larger than cluster size.
            cluster_sample_sizes = np.clip(cluster_sample_sizes, 1, cluster_sizes)

            cluster_sample_indices = []
            for cluster_value, cluster_sample_size in zip(cluster_values, cluster_sample_sizes):
                cluster_sample_indices.append(
                    np.random.choice(np.where(cluster_label == cluster_value)[0], cluster_sample_size, replace=False))
            sample_indices = np.concatenate(cluster_sample_indices)
        else:
            sample_indices = np.random.choice(N, sample_size, replace=False)

        return sample_indices

    # Select by quantile
    if quantile is not None:
        def inner_average(x):
            # TODO: Use weighted average.
            return np.average(x)

        v_inner_average = np.vectorize(inner_average)
        feature_y_average = v_inner_average(partial[:, 1])
        quantile_values = np.quantile(feature_y_average, quantile, interpolation='nearest')

        quantile_indices = []
        for quantile_value in quantile_values:
            # Select the index of value where they first appear.
            quantile_index = np.where(feature_y_average == quantile_value)[0][0]
            quantile_indices.append(quantile_index)

        return quantile_indices


def sample_suffix(sample_size=None, quantile=None):
    if sample_size is not None:
        suffix = f'_Sample{sample_size}'
    elif quantile is not None:
        suffix = '_Q' + ';'.join(map(str, quantile))
    else:
        suffix = ''

    return suffix


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

        # Padding according to the cluster label length.
        plt.title(f'SPPDP of Typical Cluster in Feature {feature_index + 1}', pad=10 + 15 * math.ceil(len(inner_vector) / 5))
        plt.legend(
            loc='lower center', bbox_to_anchor=(0.5, 1), ncol=5,
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
        partial, temporal, cluster_label=None,
        sample_size=None, quantile=None,
        feature_name=None
):
    sample_indices = sample_partial(partial, sample_size, quantile, cluster_label)
    local_count = len(sample_indices)

    # Generate index label
    index_label = np.arange(len(partial))
    index_label = index_label[sample_indices]

    # Select partial, temporal and cluster.
    partial = partial[sample_indices]
    temporal = temporal[sample_indices]
    if cluster_label is not None:
        cluster_label = cluster_label[sample_indices]

    # Trace name and show_legend control
    if cluster_label is not None:
        colors = vector_to_color(cluster_label)

        def inner_naming(label):
            return f'Cluster {label}'

        v_naming = np.vectorize(inner_naming)
        names = v_naming(cluster_label)

        def inner_unique(label):
            _, first_index = np.unique(label, return_index=True)
            first_vector = np.zeros_like(label, dtype=bool)
            first_vector[first_index] = True
            return first_vector

        show_vector = np.apply_along_axis(inner_unique, -1, cluster_label)

    else:
        colors = vector_to_color(temporal)
        names = np.empty_like(temporal, dtype=object)
        show_vector = np.zeros_like(temporal, dtype=bool)

    # Each local corresponds to each trace
    trace_list = []
    for local_index in range(local_count):
        x = partial[local_index, 0]
        y = partial[local_index, 1]
        trace = go.Scatter3d(
            y=x, z=y,
            x=np.tile(temporal[local_index], len(x)),
            text=y,
            mode='lines',
            line=dict(
                # Receive Color String
                color=colors[local_index],
                width=2,
            ),
            name=names[local_index],
            legendgroup=names[local_index],
            showlegend=bool(show_vector[local_index]),
            hovertemplate=
            '<b>X Value</b>: %{y} <br />' +
            '<b>Time Slice</b>: %{x}  <br />' +
            f'<b>Index</b>: {index_label[local_index]}  <br />' +
            '<b>Partial Value</b>: %{z}  <br />'

        )
        trace_list.append(trace)

    fig = go.Figure(data=trace_list)
    if feature_name:
        title = f'SPPDP of Feature {feature_name}'
    else:
        title = 'SPPDP'

    fig.update_layout(
        title={
            'text': title,
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
    y_max = np.max([np.max(y) for y in partial[:, 1]])
    y_min = np.min([np.min(y) for y in partial[:, 1]])

    x_max = np.max([np.max(x) for x in partial[:, 0]])
    x_min = np.min([np.min(x) for x in partial[:, 0]])

    fig.update_scenes(
        xaxis_title='Time Slice',
        xaxis_range=range_margin(vector=temporal),
        yaxis_title='Independent / X value',
        yaxis_range=range_margin(value_min=x_min, value_max=x_max),
        zaxis_title='Dependent / Partial Value',
        zaxis_range=range_margin(value_min=y_min, value_max=y_max),
    )

    return fig


def partials_plot_3d(
        feature_partial, temporal, cluster_labels=None,
        sample_size=None, quantile=None, feature_names=None
):
    """

    Args:
        feature_partial ():
        temporal ():
        cluster_labels (): Shape(N,) or Shape(Feature, N)
        sample_size ():
        quantile ():
        feature_names ():

    Returns:

    """

    feature_count = len(feature_partial)

    # Feature cluster or Integrated cluster.
    if cluster_labels is not None:
        # If Shape(N,). Else Shape(Feature, N).
        if len(cluster_labels.shape) == 1:
            cluster_labels = np.tile(cluster_labels.reshape(1, -1), (feature_count, 1))

    # Iterate each feature
    fig_list = []
    for feature_index in range(feature_count):
        fig = partial_plot_3d(
            partial=feature_partial[feature_index],
            temporal=temporal,
            cluster_label=cluster_labels[feature_index] if cluster_labels is not None else None,
            sample_size=sample_size,
            quantile=quantile,
            feature_name=feature_names[feature_index] if feature_names is not None else None,
        )

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

        plot=False, select_clusters=False,
        plot_title='Condensed trees', plot_filename='CondensedTrees.png', plot_folder=default_folder,
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
        plot:
        plot_filename:
        plot_title:
        plot_folder:

    Returns:

    """

    from hdbscan import HDBSCAN
    from umap import UMAP

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

    # TODO: Stable Reproducible result.
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

    if plot:
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

    # Individual feature cluster
    feature_count = features_distance.shape[0]

    # Shape(Feature, N, 2)
    features_embedding = []
    # Shape(Feature, N)
    features_cluster_label = []
    for feature_index in range(feature_count):
        cluster_embedding, cluster_label, _ = partial_cluster(
            distance=features_distance[feature_index],
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

    return features_embedding, features_cluster_label, features_distance


def choose_cluster_typical(embedding, cluster_vector):
    """
    Return the index of typical items for each cluster.
    The typical item of a cluster is the centre of the cluster,
    which has the minimal summation of distance to others in the same cluster.

    Args:
        embedding ():
        cluster_vector ():

    Returns: List of index of typical items. The length of the list is the number of clusters.

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
        embedding, cluster, temporal_vector, feature_name
):
    """
    2D Embedding plot colored by cluster.

    Args:
        embedding (np.ndarray): Shape(N, 2)
        cluster (): Shape(N,)
        temporal_vector (): Shape(N,)
        feature_name ():
        filename ():
        folder ():

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

    title = f'Low dimension embedding'
    if feature_name:
        title += f' of {feature_name}'

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

    return fig


def compass_plot(
        cluster_fig, partial_fig, embedding_fig,
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

    return fig


def partial_compound_plot(
        geo_vector, temporal_vector, feature_partial,
        embedding, cluster_label,
        sample_size=None, quantile=None,
        feature_names=None, folder=default_folder,
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
        embedding (): Shape(Feature, N, 2) for individual cluster, Shape(N, 2) for merged cluster.
        cluster_label (): Shape(Feature, N) for individual cluster, Shape(N,) for merged cluster.
        sample_size ():
        quantile:
        feature_names (): Shape(Feature)
        folder ():


    Returns:

    """

    # TODO: Add hover highlight.

    feature_count = len(feature_partial)

    partial_figs = partials_plot_3d(
        feature_partial, temporal_vector, cluster_label,
        sample_size=sample_size, quantile=quantile, feature_names=feature_names
    )

    if len(embedding.shape) == 2 and len(cluster_label.shape) == 1:
        embedding_fig = embedding_plot(
            embedding, cluster_label, temporal_vector,
            f'total features',
        )
        embedding_figs = [embedding_fig for _ in range(feature_count)]

        cluster_fig = scatter_3d(
            geo_vector, temporal_vector, cluster_label,
            f'Merged Spatio-temporal Cluster Plot', 'Cluster Label',
            filename=f'Cluster_Merged', is_cluster=True, folder=folder)
        cluster_figs = [cluster_fig for _ in range(feature_count)]
    else:
        embedding_figs = [
            embedding_plot(
                embedding[feature_index], cluster_label[feature_index], temporal_vector,
                f'Feature {feature_index + 1} {feature_names[feature_index] if feature_names is not None else ""}',
            )
            for feature_index in range(feature_count)
        ]

        cluster_figs = [
            scatter_3d(
                geo_vector, temporal_vector, cluster_label[feature_index],
                f'Spatio-temporal Cluster Plot of Feature {feature_index + 1} {feature_names[feature_index] if feature_names is not None else ""}',
                'Cluster Label',
                filename=f'Cluster_{feature_index + 1}', is_cluster=True, folder=folder)
            for feature_index in range(feature_count)
        ]

    compass_figs = [
        compass_plot(
            cluster_figs[feature_index], partial_figs[feature_index], embedding_figs[feature_index],
        )
        for feature_index in range(feature_count)
    ]

    return partial_figs, embedding_figs, cluster_figs, compass_figs


if __name__ == '__main__':
    sample_partial()
