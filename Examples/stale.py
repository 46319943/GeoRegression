import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from georegression.local_ale import weighted_ale
from georegression.simulation.simulation_utils import coefficient_wrapper
from georegression.visualize.ale import plot_ale
from georegression.weight_model import WeightModel

from georegression.simulation.simulation_for_ale import (
    coef_manual_gau,
    f_interact,
    generate_sample,
)

# Font family
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["font.weight"] = "bold"
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15

f = f_interact
coef_func = coef_manual_gau
x2_coef = coefficient_wrapper(partial(np.multiply, 3), coef_func())


def draw_graph():
    X, y, points, f, coef = generate_sample()
    distance_measure = "euclidean"
    kernel_type = "bisquare"
    neighbour_count = 0.05

    model = WeightModel(
        RandomForestRegressor(n_estimators=50),
        distance_measure,
        kernel_type,
        neighbour_count=neighbour_count,
        cache_data=True,
        cache_estimator=True,
    )
    model.fit(X, y, [points])
    print("GRF:", model.llocv_score_)

    feature_index = 0

    for local_index in range(model.N):
        estimator = model.local_estimator_list[local_index]
        neighbour_mask = model.neighbour_matrix_[local_index]
        neighbour_weight = model.weight_matrix_[local_index][neighbour_mask]
        X_local = model.X[neighbour_mask]
        ale_result = weighted_ale(
            X_local, feature_index, estimator.predict, neighbour_weight
        )

        fval, ale = ale_result

        x_neighbour = X[model.neighbour_matrix_[local_index], feature_index]
        y_neighbour = y[model.neighbour_matrix_[local_index]]
        weight_neighbour = model.weight_matrix_[
            local_index, model.neighbour_matrix_[local_index]
        ]

        fig = plot_ale(fval, ale, x_neighbour)
        fig.set_size_inches(10, 6)
        ax1 = fig.get_axes()[0]
        ax2 = fig.get_axes()[1]

        ax1.set_xlabel("Feature value", fontweight="bold")
        ax1.set_ylabel("Function value", fontweight="bold")
        ax2.set_ylabel("Density", fontweight="bold")

        scatter = ax1.scatter(x_neighbour, y_neighbour, c=weight_neighbour)
        ax1.scatter(
            X[local_index, feature_index], y[local_index], c="red", label="Local point"
        )
        cbar = fig.colorbar(scatter, ax=ax1, label="Weight", pad=0.1)
        cbar.set_label("Weight", weight="bold")
        cbar.ax.tick_params(labelsize=15)

        # Add legend
        handles, labels = ax1.get_legend_handles_labels()
        handles.append(scatter)
        labels.append("Weight")
        ax1.legend(handles, labels, fontsize=15)

        folder_name = "Plot/LocalAle_BigFont"
        os.makedirs(folder_name, exist_ok=True)
        plt.savefig(f"{folder_name}/{local_index}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    draw_graph()
