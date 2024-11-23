# Written by Yixiao Ge

import warnings

import faiss
import torch

from ..utils import to_numpy, to_torch

__all__ = ["label_generator_kmeans"]


@torch.no_grad()
def label_generator_kmeans(features, num_classes=500, cuda=True):

    assert num_classes, "num_classes for kmeans is null"

    # k-means cluster by faiss
    cluster = faiss.Kmeans(
        features.size(-1), num_classes, niter=300, verbose=True, gpu=cuda
    )

    cluster.train(to_numpy(features))

    _, labels = cluster.index.search(to_numpy(features), 1)
    labels = labels.reshape(-1)

    centers = to_torch(cluster.centroids).float()
    # labels = to_torch(labels).long()

    # k-means does not have outlier points
    assert not (-1 in labels)

    return labels, centers, num_classes, None
