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
    def select_feature_LR_wrapper(n_para, x, y, model_type):

        if model_type == 'n':
            general_simple = LogisticRegression()
            general_simple.fit(x, y)
            original_model_paras = general_simple.coef_[0]
            index = np.argsort(abs(original_model_paras))[::-1]
            list_of_select_features = index[:n_para]
            new_x = x[:,list_of_select_features]
            LR_refit = LogisticRegression()
            LR_refit.fit(new_x, y)
            return np.concatenate((LR_refit.coef_[0], LR_refit.intercept_)), 1 - LR_refit.score(new_x, y)


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
