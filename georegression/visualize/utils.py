import numpy as np
from matplotlib import colors, cm


def color_to_str(color_vector):
    def inner_stringify(color):
        return np.array(
            f'rgba({int(color[0] * 255)},{int(color[1] * 255)},{int(color[2] * 255)},{int(color[3] * 255)})',
            dtype=object
        )

    return np.apply_along_axis(inner_stringify, -1, color_vector)


def vector_to_color(vector, stringify=True, colormap=None):
    norm = colors.Normalize()
    color = cm.get_cmap(colormap)(norm(vector))
    if not stringify:
        return color
    return color_to_str(color)
