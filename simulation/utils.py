import numpy as np
import matplotlib.pyplot as plt
from numpy.random import laplace
class utils:

    def subSampling(point, number_of_samples, decay, blackbox_model=None):
        # the prior distribution uses exponential
        # assume the point are scaled in
        if point == None or len(point) < 1:
            raise Exception('the subsampled point is empty')
        if all( e > 1 and e < 0 for e in point):
            raise Exception('the subsampled point is not within one and zero')
        point_dimension = len(point)
        samples = np.zeros((number_of_samples,len(point)))
        for p in range(number_of_samples):
            within_range = False
            while not within_range:
                cur_sample = laplace(point, decay*np.ones((point_dimension,)) , point_dimension)
                if all( e <= 1 and e >= 0 for e in cur_sample):
                    within_range = True
            samples[p] = cur_sample
        return samples

    # def rescale_to_one_zero(ret):
    #     def rescale(vector):
    #         max = np.max(vector)
    #         min = np.min(vector)
    #         return (vector-min)/(max-min)
    #     return  np.apply_along_axis(rescale,0, ret)
if __name__ == "__main__":
    point = [0.1,0.5]
    subsamples = utils.subSampling(point,10,1)
    print(subsamples)
    plt.plot(point[0],'ro')
    plt.plot(point[1], 'bo')
    plt.scatter(np.zeros(10),subsamples[:,0],c = 'y')
    plt.scatter(np.zeros(10),subsamples[:,1],c = 'g')
    plt.show()