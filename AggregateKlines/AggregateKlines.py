from AggregateKlines.node import Node
from AggregateKlines.similarity_pairs import SimilarityPairs
import numpy as np
from AggregateKlines.utilities import utils
from simulation.simulate_class import Simulator

class AggregateKlines(object):

    def _initialize_clusters(self, n):
        clusters = []
        for i in range(n):
            clusters.append(Node(i))
        return clusters

    def _update_centroid(self, cluster):
        self.cluster.update_centroid(np.mean(self.X(cluster.get_index()), axis=0))

    def _get_coef_helper(self, centroid, n_s):
        X, Y = self.sample_function(centroid, n_s)
        coefs, _ = self.k_lin_clf(X, Y)
        return coefs

    def _set_coefs(self, cluster, n_s):
        coefs = self._get_coef_helper(cluster.get_centroid(), n_s)
        cluster.update_coefs(coefs)

    def _initialize_all_similarities(self):
        for i in range(len(self.clusters)):
            for j in range(i, len(self.clusters)):
                cluster1 = self.clusters[i]
                cluster2 = self.clusters[j]
                similarity_score = np.norm(cluster1-cluster2)
                self.similarity_pairs.set_similarity_score(cluster1, cluster2, similarity_score)

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
            self._set_coefs(self.clusters[i], self.k_sparse)
        self._initialize_all_similarities()

    def aggregate_one_round(self):
        key_tuple = self.similarity_pairs.find_minimal_pair()
        self.aggregate(key_tuple[0], key_tuple[1])

    def aggregate(self, cluster1, cluster2):
        aggregate_index = cluster1.get_index()
        aggregate_index.extend(cluster2.get_index())
        new_cluster = Node(aggregate_index)
        self._update_centroid(new_cluster)
        self._set_coefs(new_cluster, self.K_sparse)
        self.clusters.remove(cluster1)
        self.clusters.remove(cluster2)
        self.similarity_pairs.remove_cluster(cluster1)
        self.similarity_pairs.remove_cluster(cluster2)
        for cluster in self.clusters:
            self.similarity_pairs[(cluster, new_cluster)] = cluster.get_coefs()-new_cluster.get_coefs()


if __name__ == "__main__":

    n_d = 2000
    n_f = 12
    ls_split_features = [[1], [2, -1], [-1, 4]]
    ls_split_values = [[0.2], [.5, None], [None, 0.7]]
    leaf_feature_ls = [[1, 4], [3, 5], [2, 3], [1, 2]]
    leaf_beta_ls = [np.array([0, -0.5, 0.2]), np.array([-0.4, 0.2, 0.4]), np.array([-0.2, 0.1, 0.2]),
                    np.array([0.4, -0.5, -1])]
    simulator = Simulator(n_d, n_f, ls_split_features, ls_split_values, leaf_feature_ls, leaf_beta_ls)
    X, Y = simulator.get_simulated_data()
    simulator = AggregateKlines(X, Y, k_lin_clf, sample_function, black_box_model, sparse_K)


