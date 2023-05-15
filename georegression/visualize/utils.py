import numpy as np
from matplotlib import colors, cm
from plotly import graph_objects as go


def color_to_str(color_vector):
    def inner_stringify(color):
        return np.array(
            f'rgba({int(color[0] * 255)},{int(color[1] * 255)},{int(color[2] * 255)},{int(color[3] * 255)})',
            dtype=object
        )

    return np.apply_along_axis(inner_stringify, -1, color_vector)


def vector_to_color(vector, stringify=True, colormap='viridis'):
    """
    Set style template will change the default colormap.

    Args:
        vector ():
        stringify ():
        colormap ():

    Returns:

    """
    vector_normed = (vector - np.min(vector, axis=-1, keepdims=True)) / (
            np.max(vector, axis=-1, keepdims=True) - np.min(vector, axis=-1, keepdims=True))
    color = cm.get_cmap(colormap)(vector_normed)
    if not stringify:
        return color
    return color_to_str(color)


def range_margin(vector=None, value_min=None, value_max=None, margin=0.05):
    if vector is None and (value_min is None or value_max is None):
        raise Exception('Invalid parameter')

    if vector is not None:
        value_min = np.min(vector)
        value_max = np.max(vector)

    interval = value_max - value_min

    return [value_min - interval * margin, value_max + interval * margin]


def aspect_ratio_scale(fig: go.Figure, scale: float = 0.2):
    x_aspect = fig.layout.scene.aspectratio.x
    y_aspect = fig.layout.scene.aspectratio.y
    z_aspect = fig.layout.scene.aspectratio.z

    fig.update_layout(
        scene_aspectratio={
            "x": x_aspect * scale,
            "y": y_aspect * scale,
            "z": z_aspect * scale,
        },
    )

    return fig


def camera_position(fig: go.Figure, rotation=-70, pitch=40):
    # From rotation to x y using sin and cos
    x = np.cos(rotation * np.pi / 180)
    y = np.sin(rotation * np.pi / 180)

    # From pitch to z using sin
    z = np.sin(pitch * np.pi / 180)
    camera = dict(
        eye=dict(x=x, y=y, z=z)
    )
    fig.update_layout(scene_camera=camera)

    return fig
