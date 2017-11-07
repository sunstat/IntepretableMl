import numpy as np
import matplotlib.pyplot as plt
from numpy.random import laplace
import sklearn
from sklearn import datasets
from sklearn.linear_model import LogisticRegression


class utils(object):

    @staticmethod
    def sub_sampling(point, number_of_samples, decay, blackbox_model=None):
        # the prior distribution uses exponential
        # assume the point are scaled in
        if not point or len(point) < 1:
            raise Exception('the subsampled point is empty')
        if all([e > 1 and e < 0 for e in point]):
            raise Exception('the subsampled point is not within one and zero')
        point_dimension = len(point)
        samples = np.zeros((number_of_samples, len(point)))
        for p in range(number_of_samples):
            within_range = False
            while not within_range:
                cur_sample = laplace(point, decay*np.ones((point_dimension,)) , point_dimension)
                if all( e <= 1 and e >= 0 for e in cur_sample):
                    within_range = True
            samples[p] = cur_sample
        return samples

    @staticmethod
    def select_feature_LR_wrapper(N_para, x, y):
        # fit lr without any regularizer to get the coefficient
        # only deal with binary classification !!!
        general_simple = LogisticRegression()
        general_simple.fit(x, y)
        # print('not_sparse_model',general_simple.score(x,y))
        original_model_paras = general_simple.coef_

        LR = LogisticRegression()
        number_of_class = len(original_model_paras)
        original_para_size = len(original_model_paras[0])
        list_of_select_features = []
        model_select_para = []
        model_select_intercept = []
        for model_para in original_model_paras:
            index = np.argsort(abs(model_para))[::-1]
            list_of_select_features.append(index[:N_para])
        # now refit the model with selected feature
        # assume the class is labeled as 0,1,2, ...
        for i in range(number_of_class):
            # transfer the original data into one-vs-rest
            y_i = [1 if y_j == i else 0 for y_j in y]
            x_i = x[:, list_of_select_features[i]]
            LR.fit(x_i, y_i)
            # print(LR.decision_function(x_i[0]))
            para = np.zeros(original_para_size)
            para[list_of_select_features[i]] = LR.coef_[0]
            model_select_para.append(para)
            model_select_intercept.append(LR.intercept_[0])
        LR_re = LogisticRegression()
        LR_re.coef_ = np.array(model_select_para)
        LR_re.intercept_ = np.array(model_select_intercept)
        LR_re.classes_ = LR.classes_
        # print('new_model',LR_re.score(x,y_i))
        return np.concatenate((LR_re.coef_[0], LR_re.intercept_)), 1 - LR_re.score(x, y_i)
            # def rescale_to_one_zero(ret):
    #     def rescale(vector):
    #         max = np.max(vector)
    #         min = np.min(vector)
    #         return (vector-min)/(max-min)
    #     return  np.apply_along_axis(rescale,0, ret)

if __name__ == "__main__":

    # test sbusampling function
    point = [0.1,0.5]
    subsamples = utils.subSampling(point,10,1)
    print(subsamples)
    plt.plot(point[0],'ro')
    plt.plot(point[1], 'bo')
    plt.scatter(np.zeros(10),subsamples[:,0],c = 'y')
    plt.scatter(np.zeros(10),subsamples[:,1],c = 'g')
    #plt.show()

    # test wrapper function
    x, y = datasets.load_digits(n_class=2, return_X_y=True)
    coef,error = utils.select_feature_LR_wrapper(3, x, y)
    print('coef:', coef)
    print('error:', error)
