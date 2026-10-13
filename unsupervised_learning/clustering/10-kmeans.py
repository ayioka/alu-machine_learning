#!/usr/bin/env python3
"""Perform K-means clustering using scikit-learn."""
import sklearn.cluster


def kmeans(X, k):
    """Performs K-means on a dataset using sklearn."""
    model = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = model.cluster_centers_
    clss = model.labels_
    return C, clss
