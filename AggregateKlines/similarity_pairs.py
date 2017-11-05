import numpy as np
import sys

class SimilarityPairs(object):

    def __init__(self, list_objects):
        '''
        :param list_objects:
        '''
        self.similarity_pairs = {}
        for i in range(len(list_objects)):
            for j in range(i, len(list_objects)):
                self.similarity_pairs[(list_objects[i], list_objects[j])] = float("inf")

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







