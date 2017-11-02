from simulation.node import Node
import numpy as np
import random
from sklearn.model_selection import train_test_split
import math
import numpy.random as rd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class DeterministicDecisonTree(object):

    def __init__(self, X, Y, ls_split_features, ls_split_values, leaf_feature_ls, leaf_beta_ls):

        '''
        :param X: data, numpy array
        :param Y: labels, numpy array
        :param ls_split_featuress: the size of this list is the number of layers in the tree
        :param ls_split_values: value used to split
        :param leaf_feature_ls: features used for each leaf to fit a logistic regression
        :param leaf_beta_ls: betas used for each leaf to generate labels
        '''

        self.X = X
        self.Y = Y
        self.ls_split_features = ls_split_features
        self.ls_split_values = ls_split_values
        self.leaf_feature_ls = leaf_feature_ls
        self.leaf_beta_ls = leaf_beta_ls
        self.nodes = []
        self.leafs = []
        self.root = Node(np.array(range(len(Y))))
        self.nodes.append(self.root)
        self.leafs.append(self.root)
        self.finish_splitting = False

    def split(self, node, s_f, s_v):

        '''
        :param node: node to be splitted
        :param i_f: the index of feature to determine this split
        :param s_v:
        :return:
        '''

        indexes = node.get_index()
        values = self.X[indexes,s_f]
        left_indexes = indexes[np.where(values <= s_v)]
        right_indexes = indexes[np.where(values > s_v)]
        left_node = Node(left_indexes)
        right_node = Node(right_indexes)
        node.set_kids(left_node, right_node)
        self.leafs.remove(node)
        self.leafs.append(left_node)
        self.leafs.append(right_node)
        self.nodes.append(left_node)
        self.nodes.append(right_node)

    def execute_splits(self):
        layers = len(self.ls_split_features)
        for layer_iter in layers:
            for leaf_iter in range(len(self.leafs)):
                leaf = self.leafs[leaf_iter]
                s_f = self.ls_split_features[layer_iter][leaf_iter]
                s_v = self.ls_split_values[layer_iter][leaf_iter]
                self.split(leaf, s_f, s_v)
        self.finish_splitting = True


    def assign_labels(self):
        '''
        assign labels after finishing split
        :return: None
        '''

        if not self.finish_splitting:
            print("not finish splitting yet, please do splitting first by calling execute_splits")
            return
        for l_i in range(len(self.leafs)):
            leaf = self.leafs[l_i]
            ls_f = self.leaf_feature_ls[l_i]
            beta = self.leaf_beta_ls
            indexes = leaf.get_index()
            values = self.X[indexes, :][:,ls_f]
            d_v = np.dot(values, beta[1:])+beta[0]
            self.Y[indexes[np.where(d_v>=0)]] = 1
            self.Y[indexes[np.where(d_v<0)]] = 0

    def get_data(self):

        return (self.X, self.Y)


    def refresh(self):











