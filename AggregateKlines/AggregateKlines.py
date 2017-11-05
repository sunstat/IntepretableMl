from AggregateKlines.node import Node
from AggregateKlines.similarity_pairs import SimilarityPairs
import numpy as np
from AggregateKlines.utilities import utils


class AggregateKlines(object):

    def _initialize_clusters(self, n):
        clusters = []
        for i in range(n):
            clusters.append(Node(i))
        return clusters

    def _update_centroid(self, cluster):
        self.cluster.set_centroid(self.X(cluster.get_index(), axis=1))

    def _get_coef_helper(self, centroid, n_s):
        X, Y = self.sample_function(centroid, n_s)
        coef, _ = self.k_lin_clf(X, Y)
        return coef

    def _set_beta(self, cluster, n_s):
        beta = self._get_coef_helper(cluster.centroid, n_s)
        cluster.update_beta(beta)

    def _set_all_similarities(self):
        for i in range(len(self.clusters)):
            for j in range(i, len(self.clusters)):
                cluster1 = self.clusters[i]
                cluster2 = self.clusters[j]
                similarity_score = np.norm(cluster1-cluster2)
                self.similarity_pairs.set_similarity_score(cluster1,cluster2, similarity_score)

    def __init__(self, X, Y, k_lin_clf, sample_function, black_box_model, sparse_K):
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
        self.similarity_pairs = SimilarityPairs(self.clusters)
        self.black_box_model = black_box_model
        self.sparse_K = sparse_K
        for i in range(self.clusters):
            self._set_beta(self.clusters[i], self.k_sparse)
        self._set_all_similarities()

    def aggregate_one_round(self):
        key_tuple = self.similarity_pairs.find_minimal_pair()
        

    def aggregate(self, cluster1, cluster2, new_beta):
        aggregate_index = cluster1.get_index()
        aggregate_index.extend(cluster2.get_index())
        new_cluster = Node(aggregate_index, new_beta)
        self._update_centroid(new_cluster)
        self.clusters.remove(cluster1)
        self.clusters.remove(cluster2)
        self.clusters.append(new_cluster)
        self.similarity_pairs.remove_cluster(cluster1)
        self.similarity_pairs.remove_cluster(cluster2)

if __name__ == "__main__":

