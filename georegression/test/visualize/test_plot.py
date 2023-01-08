import pandas as pd
import plotly.graph_objects as go
import numpy as np
from matplotlib import cm
import plotly.express as px


def value_to_rgb_str(value, colormap_name='viridis'):
    colormap = cm.get_cmap(colormap_name)
    rgba_tuple = colormap(value)
    return f'rgb({int(rgba_tuple[0] * 255)}, {int(rgba_tuple[1] * 255)}, {int(rgba_tuple[2] * 255)})'


def value_list_to_rgb_str_list(value_list):
    return [
        value_to_rgb_str(value)
        for value in value_list
    ]


colors = px.colors.sequential.RdBu


def test_mesh_3d():
    fig = go.Figure(data=[
        # Point
        go.Scatter3d(x=[0, 1, 1, 0], y=[0, 0, 1, 1], z=[0, 0, 0, 0], hovertext=[1, 2, 3, 4],
                     marker={
                         # 'colorscale': px.colors.qualitative.Plotly,
                         # 'colorscale': value_list_to_rgb_str_list([0, 0.25, 0.25, 0.88]),
                         'colorscale': colors,

                         'colorbar': {
                             'x': 1,
                             'tickformat': '.4s',
                             # 'tickvals': value_list_to_rgb_str_list([0, 0.25, 0.25, 0.88])
                             # 'tickvals': [0, 1]
                         },
                         # 'color': value_list_to_rgb_str_list([0, 0.25, 0.25, 0.88])
                         'color': [0, 0.25, 0.25, 0.88]
                     }),
        # Line
        go.Scatter3d(x=[0, 1, 1, 0], y=[0, 0, 1, 1], z=[0, 1, 1, 0], hovertext=[1, 2, 3, 4],
                     mode='lines',
                     line={
                         # 'colorbar': {'x': 1.1},
                         # 'color': [1, 2, 3, 4]
                     }),
        # Bottom surface
        go.Mesh3d(x=[0, 1, 1, 0], y=[0, 0, 1, 1], z=[0, 0, 0, 0], color='rgb(0, 255, 0)', hoverinfo='skip',
                  hovertemplate=None, colorbar={'x': 1.1}),
        # Top surface
        go.Mesh3d(x=[0, 1, 1, 0], y=[0, 0, 1, 1], z=[1] * 4, color='rgb(255, 255, 0)', hoverinfo='skip',
                  hovertemplate=None)
    ])
    fig.update_layout(
        # Set axis ratio
        scene_aspectmode='manual',
        scene_aspectratio=dict(x=1, y=1, z=3),
        # Clear margin
        # automargin=True
        margin=dict(l=0, r=0, t=50, b=0, pad=0),
        hovermode='x',
        # hoverdistance=-1
        spikedistance=1
    )

    fig.update_layout(
        title={
            'text': f"Temporal Partial Dependency of Feature",
            'y': 0.95,  # new
            'x': 0.4,
            'xanchor': 'center',
            'yanchor': 'top'  # new
        },
        scene=dict(
            xaxis_title='X Axis Title',
            yaxis_title='Y Axis Title',
            zaxis_title='Z Axis Title',
        ),
        legend_title="Cluster Legend",
        font=dict(
            size=18,
            color="RebeccaPurple"
        ))

    # fig.update_xaxes(automargin=True)
    # fig.update_yaxes(automargin=True)

    fig2 = px.scatter_3d(x=[0, 1, 1, 0], y=[0, 0, 1, 1], z=[0, 0, 0, 0],
                         color=np.array([0, 0.25, 0.25, 0.88]).astype(str))
    fig.add_traces(fig2.data)

    fig.write_html(f'test_plot.html')


def test_discrete():
    fig = px.scatter_3d(x=[0, 1, 1, 0], y=[0, 0, 1, 1], z=[0, 0, 0, 0],
                        color=['T1', 'T2', 'T3', 'T4'])
    fig.write_html(f'test_plot2.html')


def test_lengend():
    import plotly.express as px
    df = px.data.gapminder()
    fig = px.line(df, y="lifeExp", x="year", color="continent", line_group="country",
                  line_shape="spline", render_mode="svg",
                  color_discrete_sequence=px.colors.qualitative.G10,
                  title="Built-in G10 color sequence")

    fig.show()


def test_surface_colorbar():
    fig = go.Figure([
        go.Mesh3d(
            x=[0, 1, 1, 0], y=[0, 0, 1, 1], z=[0, 0, 0, 0], color='rgb(0, 255, 0)', hoverinfo='skip',
            hovertemplate=None,
            name='Surface Group',
            legendgroup='surface group',
            showlegend=True,
            showscale=True,
            colorbar={'x': 1.1}
        ),
        go.Mesh3d(
            x=[0, 1, 1, 0], y=[0, 0, 1, 1], z=[1, 1, 1, 1], hoverinfo='skip',
            hovertemplate=None,
            name='surface2',
            legendgroup='surface group',
            showlegend=False,
            # colorbar={'x': 1.1}
        )
    ])

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                backgroundcolor="rgb(200, 200, 230)",
                gridcolor="white",
            ),
        )
    )

    fig.write_html(f'test_plot.html')


def test_axes_ratio():
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3],
        y=[0, 0, 1, 1, 3, 3, 2, 2, 3, 3, 1, 1, 0, 0]
    ))

    fig.update_layout(
        width=800,
        height=500,
        title="fixed-ratio axes"
    )
    fig.update_xaxes(
        range=(-0.5, 3.5),
        constrain='domain'
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    fig.show()

if __name__ == '__main__':
    pass
