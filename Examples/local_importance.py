import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from georegression.weight_model import WeightModel
from georegression.simulation.simulation_for_importance import coef_auto_gau_weak, coef_auto_gau_strong, f_square_2, generate_sample


def draw_graph():
    X, y, points = generate_sample(
        count=5000, f=f_square_2, coef_func=[coef_auto_gau_strong, coef_auto_gau_weak], random_seed=1,
        plot=True
    )
    distance_measure = "euclidean"
    kernel_type = "bisquare"
    neighbour_count = 0.02

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

    importance_global = model.importance_score_global()
    print("Global Importance Score: ", importance_global)

    importance_local = model.importance_score_local()
    print("Local Importance Socre Shape: ", importance_local.shape)

    # Plot the local importance
    for i in range(importance_local.shape[1]):
        fig = plt.figure()
        scatter = plt.scatter(
            points[:, 0], points[:, 1], c=importance_local[:, i], cmap="viridis"
        )
        fig.colorbar(scatter)
        fig.savefig(f"Plot/Local_importance_{i}.png")


if __name__ == "__main__":
    draw_graph()
