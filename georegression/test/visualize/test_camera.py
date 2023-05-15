import pandas as pd
import plotly.graph_objects as go
import numpy as np
from matplotlib import cm
import plotly.express as px

colors = px.colors.sequential.RdBu
def test_camera_static():
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



    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0.25, y=13.25, z=13.25),
        projection_type="orthographic"
    )

    fig.update_layout(scene_camera=camera)

    # Get and set the aspect ratio of the scene
    x_aspect = fig.layout.scene.aspectratio.x
    y_aspect = fig.layout.scene.aspectratio.y
    z_aspect = fig.layout.scene.aspectratio.z

    fig.update_layout(
        scene_aspectratio={
            "x": x_aspect * 0.1,
            "y": y_aspect * 0.1,
            "z": z_aspect * 0.1,
        },
    )

    aspectratio = dict(x=x_aspect, y=y_aspect, z=z_aspect),

    fig.write_html(f'test_plot.html', include_plotlyjs='cdn')
    fig.write_image(f'test_plot.png', width=1080, height=1920, scale=5)

if __name__ == '__main__':
    test_camera_static()