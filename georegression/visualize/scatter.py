from pathlib import Path

import numpy as np
from plotly import express as px, graph_objects as go
from georegression.visualize.utils import vector_to_color

# Create folder for saving plot
folder = Path('Plot')
folder.mkdir(exist_ok=True)


def scatter_3d(
        geo_vector, temporal_vector, value,
        figure_title, value_name, filename=None, is_cluster=False
):
    # Shape(N, )
    x = geo_vector[:, 0]
    y = geo_vector[:, 1]
    z = temporal_vector[:, 0]

    x_min = np.min(x)
    x_max = np.max(x)
    x_interval = x_max - x_min
    y_min = np.min(y)
    y_max = np.max(y)
    y_interval = y_max - y_min
    z_min = np.min(z)
    z_max = np.max(z)
    z_interval = z_max - z_min

    value = np.array(value)
    count = value.shape[0]
    value_min = np.min(value)
    value_max = np.max(value)
    value_interval = value_max - value_min

    # Multiple legend for cluster input.
    if is_cluster:
        # Quick way using express.
        # Maybe use the Graph Object to unify the style?
        # fig = px.scatter_3d(x=x, y=y, z=z, color=value.astype('str'))

        fig = go.Figure()
        color = vector_to_color(value)

        for cluster_value in np.unique(value):
            cluster_index = value == cluster_value
            fig.add_trace(
                go.Scatter3d(
                    x=x[cluster_index], y=y[cluster_index], z=z[cluster_index], mode='markers',
                    # Name of trace for legend display
                    name=f'Cluster {cluster_value}',
                    marker={
                        'color': color[cluster_index],
                        'size': 5,
                    },
                    text=value[cluster_index],
                    hovertemplate=
                    '<b>Time Slice</b> :' + ' %{z} <br />' +
                    f'<b>{value_name}</b> :' + ' %{text} <br />' +
                    '<extra></extra>',
                )
            )
    # Continuous value case. Single legend/trace.
    else:
        tick_value = np.quantile(value, [0, 0.25, 0.5, 0.75, 1], interpolation='nearest')
        fig = go.Figure(data=[
            # Data Point
            go.Scatter3d(
                x=x, y=y, z=z, mode='markers',
                # Name of trace for legend display
                name=f'{value_name}',
                marker={
                    'size': 5,
                    'color': value,
                    'colorscale': 'Portland',
                    # Input dict of properties to construct the ColorBar Instance
                    'colorbar': {
                        'x': 0.8,
                        'title': f'{value_name} Color Bar<br>(Quartile tick)<br> <br>',
                        'tickvals': tick_value,
                        'tickformat': '.3~f',
                    },
                },
                text=value,
                hovertemplate=
                '<b>Time Slice</b> :' + ' %{z} <br />' +
                f'<b>{value_name}</b> :' + ' %{text:.3~f} <br />' +
                '<extra></extra>'
            ),
        ])

    # TODO: Add Joint Line

    # Time Surface
    x_surface = (x_min, x_min, x_max, x_max)
    y_surface = (y_min, y_max, y_max, y_min)
    z_unique = np.unique(z)
    z_shift = z_interval / len(z_unique) * 0.08
    surface_color = vector_to_color(np.unique(z))

    fig.add_traces([
        go.Mesh3d(
            x=x_surface, y=y_surface,
            # Shift the surface down a little to avoid overlay
            z=[z - z_shift] * 4, opacity=0.3,
            color=surface_color[z_index],
            hoverinfo='skip',
            name=f'Auxiliary Surface Group',
            legendgroup='Auxiliary Surface Group',
            showlegend=True if not z_index else False
        )
        for z_index, z in enumerate(z_unique)
    ])

    # Set figure, axis and other things.

    if x_interval < y_interval:
        x_aspect = 1
        y_aspect = y_interval / x_interval
        z_aspect = y_interval * len(z_unique)
    else:
        y_aspect = 1
        x_aspect = x_interval / y_interval
        z_aspect = y_interval * len(z_unique)

    fig.update_layout(
        # Clear margin
        margin=dict(l=0, r=0, t=50, b=0, pad=0),

        # Figure title
        title={
            'text': figure_title,
            'xanchor': 'center',
            'x': 0.45,
            'yanchor': 'top',
            'y': 0.99,
        },

        # Global font
        font=dict(size=12),

        # Legend
        legend_title="Point and Surface Legend",

        # Set axis ratio
        scene_aspectmode='manual',
        scene_aspectratio=dict(x=x_aspect, y=y_aspect, z=z_aspect),

        # Axis label
        scene=dict(
            xaxis=dict(
                title='X Position',
                ticktext=['X Negative', 'X Positive'],
                tickvals=[x_min, x_max],
                range=[x_min - x_interval * 0.12, x_max + x_interval * 0.12],
                backgroundcolor="rgb(200, 200, 230)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                showspikes=False
            ),
            yaxis=dict(
                title='Y Position',
                ticktext=['Y Negative', 'Y Positive'],
                tickvals=[y_min, y_max],
                range=[y_min - y_interval * 0.12, y_max + y_interval * 0.12],
                backgroundcolor="rgb(230, 200,230)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                showspikes=False
            ),
            zaxis=dict(
                title='Temporal Slice Index',
                tickvals=z_unique,
                range=[z_min - z_interval * 0.1, z_max + z_interval * 0.1],
                backgroundcolor="rgb(230, 230,200)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                # showspikes=False
            ),
        ),
    )

    # Change Projection to Orthogonal
    fig.layout.scene.camera.projection.type = "orthographic"

    # Output to disk file
    if filename is None:
        filename = f'{figure_title}_{value_name}'

    fig.write_html(folder / f'{filename}.html')

    return fig
