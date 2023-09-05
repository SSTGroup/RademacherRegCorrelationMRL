import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score


def spectral_clustering_acc(data_points, labels, num_classes):
    """
    Compute a spectral clustering of the data_points and assign labels via majority voting
    """

    clust_labels = SpectralClustering(
        n_clusters=num_classes,
        assign_labels='kmeans',
        affinity='nearest_neighbors',
        random_state=33,
        n_init=10).fit_predict(data_points)

    prediction = np.zeros_like(clust_labels)
    for i in range(num_classes):
        ids = np.where(clust_labels == i)[0]
        prediction[ids] = np.argmax(np.bincount(labels[ids]))

    return accuracy_score(labels, prediction)
    