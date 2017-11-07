from sklearn.cluster import KMeans
from AggregateKlines.node import Node

class Kmeans(object):

    def _initialize_clusters(self, n):
        clusters = []
        for i in range(n):
            clusters.append(Node(i))
        return clusters

    def _get_coef_helper(self, centroid, n_s):
        X, Y = self.sample_function(centroid, n_s)
        coefs, _ = self.k_lin_clf(X, Y)
        return coefs

    def _set_coefs(self, cluster, n_s):
        coefs = self._get_coef_helper(cluster.get_centroid(), n_s)
        cluster.update_coefs(coefs)

    def __init__(self, X, Y, k_lin_clf, sample_function, black_box_model, sparse_K, n_clusters, n_s):
        '''
        :param X:
        :param Y:
        :param k_lin_clf:
        :param sample_function:
        '''
        self.X = X
        self.Y = Y
        self.k_lin_clf = k_lin_clf
        self.sample_function = sample_function
        self.clusters = self._initialize_clusters(len(X))
        self.black_box_model = black_box_model
        self.sparse_K = sparse_K
        self.n_clusters = n_clusters
        self.n_s = n_s

        '''
        get coeficient for every nodes
        '''
        for cluster in self.clusters:
            self._set_coefs(cluster, self.n_s)


    def KMneas_cluster(self):
        









