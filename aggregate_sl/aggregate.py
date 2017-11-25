import numpy as np
import sys
from aggregate_sl.utilities import Utils

class AggregateLocalSL(object):

    def _update_centroid(self, node):
        node.update_centroid(np.mean(self.data[node.get_index(),:], axis=0))

    def _warm_up(self, list_objects, sub_n=100):
        for node in list_objects:
            self._update_centroid(node)
            x_s = self.sampling_function(node.get_centroid(), sub_n)
            y_s = self.black_box_model(x_s)
            sub_coefs, err, min_ls = Utils.select_feature_lr_wrapper(self.k_sparse, x_s, y_s, 'bs', self.fit_intercept)
            if self.fit_intercept:
                coefs = np.zeros((self.n_feature+1, ))
                coefs[min_ls+1] = sub_coefs
            else:
                coefs = np.zeros((self.n_feature, ))
                coefs[min_ls] = sub_coefs
            node.update_coefs(coefs)



    def __init__(self, data, list_objects, sampling_function, black_box_model, k_sparse, fit_intercept = False,  k_neighbor=1):

        '''
        :param list_objects: list of objects to be clustered, each node only contains index domain
        :param sampling_function: sampling function: given a point and number of sample points, return n
        :param black_box_model : black_box_model to return label for a given point
        :param k : search nearest k neighbors to aggregate
        '''

        self.similarity_scores = {} #similarity_scores
        self.distance_map = {} #key: node; value: sorted list of pairs: (node, distance from centroid)
        self.nearest_k = k_neighbor #every time we only aggromate k nearest neighbors which default value is 1
        self.sampling_function = sampling_function
        self.black_box_model = black_box_model
        self.k_sparse = k_sparse
        self.fit_intercept = fit_intercept
        self.n_feature = len(list_objects.get_centroid())
        self.data = data


        for i in range(len(list_objects)):
            for j in range(i, len(list_objects)):
                self.similarity_scores[(list_objects[i], list_objects[j])] = float("inf")

        for i in range(len(list_objects)):
            self.distance_map[list_objects[i]] = []
            for j in range(len(list_objects)):
                if i == j:
                    continue
                self.distance_map[list_objects[i]]\
                    .append((list_objects[j], np.norm(list_objects[i].get_centroid(), list_objects[j].get_centroid())))
            self.distance_map[list_objects[i]] = sorted(self.distance_map[list_objects[i]], key=lambda x: x[1])

    def _extract_nearest_k(self, node):

        neighbors = self.distance_map[node]
        res = neighbors[:min(len(neighbors),self.nearest_k)]
        return [x[0] for x in res]


    def _contains(self, key_tuple, obj):

        '''
        :param tuple:
        :param obj:
        :return:
        '''

        for item in key_tuple:
            if item == obj:
                return True
        return False

    def remove_cluster(self, cluster):
        for key_tuple in self.similarity_pairs:
            if self._contains(key_tuple, cluster):
                del self.similarity_pairs[key_tuple]

    def add_elem(self, key_tuple, value):
        self.similarity_pairs[key_tuple] = value

    def find_minimal_pair(self):
        minimal_similarity_score = float('inf')
        final_tuple  = None
        for tuple, similarity_score in self.similarity_pairs.items():
            if minimal_similarity_score < similarity_score:
                minimal_similarity_score = similarity_score
                final_tuple = tuple
        return final_tuple

    def set_similarity_score(self, cluster1, cluster2, score):
        self.similarity_pairs[(cluster1, cluster2)] = score


