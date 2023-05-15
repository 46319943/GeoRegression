import numpy as np
from matplotlib import pyplot as plt

from georegression.visualize import default_folder


def global_importance_plot(importance_matrix, labels=None, index=True, folder_=default_folder):
    """
    Args:
        importance_matrix (np.ndarray): Shape(Feature, n_repeats).
        labels (list):
        index (): Whether add index before labels.
        folder_ ():

    """

    # Default labels if not provided.
    if labels is None:
        labels = [f'Feature {i + 1}' for i in range(importance_matrix.shape[0])]
    labels = np.array(labels)

    # Add index for labels
    if index:
        labels = [f'{i + 1}. {labels[i]}' for i in range(labels.shape[0])]
    labels = np.array(labels)

    # Sort by the mean of importance value
    importance_mean = np.mean(importance_matrix, axis=1)
    sort_index = np.argsort(importance_mean)
    importance_matrix = importance_matrix[sort_index, :]
    labels = labels[sort_index]
    importance_mean = importance_mean[sort_index]

    # Boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(importance_matrix.T, vert=False, labels=labels)

    plt.xlabel('Global Importance')
    plt.ylabel('Feature name')
    plt.title('Global Importance of Independent Features')
    plt.tight_layout()
    plt.savefig(folder_ / 'ImportanceBoxplot.png')
    plt.clf()

    # Integrate two plot into one figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharex='all')
    ax1.barh(labels, importance_mean, height=0.7)
    ax1.set_xlabel('Importance value')
    ax1.set_ylabel('Feature name')
    ax1.set_title('Mean value of feature importance')
    ax1.margins(0.02)

    ax2.boxplot(importance_matrix.T, vert=False)
    ax2.set_xlabel('Importance value')
    ax2.set_yticklabels([])
    ax2.set_title('Boxplot of feature importance')
    ax2.margins(0.02)

    fig.suptitle('Global Importance of Independent Features\n')
    fig.tight_layout()
    fig.savefig(folder_ / 'ImportancePlot.png')

# TODO: Add interaction 2D hot-plot.
