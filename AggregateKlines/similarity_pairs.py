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



