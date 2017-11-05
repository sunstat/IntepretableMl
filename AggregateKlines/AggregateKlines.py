from AggregateKlines.node import Node

class AggregateKlines(object):

    def _initialize_clusters(self, n):
        clusters = []
        for i in range(n):
            clusters.append(Node(i))
        pair_similarity = {}
        for i in range(n):
            for j in range(i+1,n):
                pair_similarity[(self.clusters[i], self.clusters[j])] = float("inf")
        return clusters

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
        self.pair_similarity =

    def aggregate(self, cluster1, cluster2, new_beta):
        aggregate_index = cluster1.get_index()
        aggregate_index.extend(cluster2.get_index())
        new_cluster = Node(aggregate_index, new_beta)
        self.clusters.remove(cluster1)
        self.clusters.remove(cluster2)
        self.clusters.append(new_cluster)
