# k-kmeans clustering with cosine distance #

import sys
import json
import os
import operator
import numpy
import math
from copy import copy
from numpy.linalg import norm
from scipy import spatial
from nltk.cluster import KMeansClusterer, euclidean_distance

def cosine_distance(u, v):
    if norm(u) == 0 or norm(v) == 0:
        return 0
    return spatial.distance.cosine(u, v)

class KmeansCosine(object):
    def __init__(self, K):
        self.num_clusters = K

    def cosine_distance(u, v):
        if norm(u) == 0 or norm(v) == 0:
            return 0
        return spatial.distance.cosine(u, v)

    def get_clusters(self, vectors):
        vectors = [numpy.array(v) for v in vectors]
        init_means=[copy(vectors[i]) for i in range(self.num_clusters)]
        clusterer = KMeansClusterer(self.num_clusters, euclidean_distance,
            initial_means=init_means, avoid_empty_clusters=True)
        clusters = clusterer.cluster(vectors, True)

        return clusters

def main():
    model = KmeansCosine(2)
    print(model.get_clusters([[1,2,3],[4,5,6],[1,2,3],[4,5,6]]))

if __name__ == "__main__":
    main()
