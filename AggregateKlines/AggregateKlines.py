from AggregateKlines.node import Node
from AggregateKlines.similarity_pairs import SimilarityPairs

class AggregateKlines(object):

    def _initialize_clusters(self, n):
        clusters = []
        for i in range(n):
            clusters.append(Node(i))
        return clusters

    def _update_centroid(self, cluster):
        self.cluster.set_centroid(self.X(cluster.get_index(), axis=1))

    def __init__(self, X, Y, k_lin_clf, sample_function):
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

    def aggregate(self, cluster1, cluster2, new_beta):
        aggregate_index = cluster1.get_index()
        aggregate_index.extend(cluster2.get_index())
        new_cluster = Node(aggregate_index, new_beta)
        self._update_centroid(new_cluster)
        self.clusters.remove(cluster1)
        self.clusters.remove(cluster2)
        self.clusters.append(new_cluster)

if __name__ == "__main__":

