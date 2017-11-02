import numpy as np
import rpy2.robjects as robjects
r = robjects.r
r.library("randomForest")

class Simulate(object):

    def __init__(self, n_d, n_f, list_split_features, list_split_values, leaf_feature_index_list, leaf_beta_list):
        '''
        :param n_d: number of data generated
        :param n_f: number of features generated
        :param list_split_features: list of features used to split data
        :param leaf_feature_index_list: leafs for each leafs
        :param leaf_beta_list: beta for logistic regression in each leaf
        '''

        #uniform generate x of size n_d x n_f

        X = np.random.uniform(0, 1, size=(n_d, n_f))







