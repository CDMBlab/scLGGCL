# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time

import faiss
import numpy as np
from PIL import Image
from PIL import ImageFile
from scipy.sparse import csr_matrix, find
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

# ImageFile.LOAD_TRUNCATED_IMAGES = True

# __all__ = ['PIC', 'Kmeans', 'cluster_assign', 'arrange_clustering']


# def preprocess_features(npdata, pca=256):
def preprocess_features(npdata, pca=64):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata = npdata.detach().cpu().numpy().astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    npdata = np.ascontiguousarray(npdata)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    # row_sums = np.linalg.norm(npdata, axis=1)
    # npdata = npdata / row_sums[:, np.newaxis]
    row_sums = np.linalg.norm(npdata, axis=1, keepdims=True)
    npdata = npdata / (row_sums + 1e-8)  # 避免除零

    return npdata

def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape
    x = np.ascontiguousarray(x)

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 50
    # res = faiss.StandardGpuResources()
    # flat_config = faiss.GpuIndexFlatConfig()
    # flat_config.useFloat16 = False
    # flat_config.device = 0

    # index = faiss.GpuIndexFlatL2(res, d, flat_config)
    index = faiss.IndexFlatL2(d)

    # perform the training
    clus.train(x, index)
    # try:
    #     clus.train(x, index)
    # except Exception as e:
    #     print(f"Faiss clustering training failed: {e}")
    #     print(f"Shape of input x: {x.shape}")
    #     print(f"Index type: {type(index)}")
    #     print(f"GPU device configuration: {flat_config.device}")
    #     return None

    _, I = index.search(x, 1)
    # losses = faiss.vector_to_array(clus.obj)
    # if verbose:
    #     print('k-means loss evolution: {0}'.format(losses))

    # return [int(n[0]) for n in I], losses[-1]  
    return I

def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]


class Kmeans(object):
    def __init__(self, k):
        self.k = k

    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(data)

        # cluster the data
        # I, loss = run_kmeans(xb, self.k, verbose)
        I = run_kmeans(xb, self.k, verbose)

        self.images_lists = [[] for i in range(self.k)]
        # for i in range(len(data)):
        #    self.images_lists[I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return I

