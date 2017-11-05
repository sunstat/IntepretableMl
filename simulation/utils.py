import numpy as np
import matplotlib.pyplot as plt
from numpy.random import exponential
class utils:

    def subSampling(point, blackbox_model, number_of_samples):
        # the prior distribution uses exponential
        # assume the point are scaled in
        if point == None or len(point) < 1:
            raise Exception('the subsampled point is empty')
        point_dimension = len(point)
        samples = np.zeros((number_of_samples,len(point)))
        for p in range(number_of_samples):
            samples[p] = exponential(point, point_dimension)
        return utils.rescale_to_one_zero(samples)

    def rescale_to_one_zero(ret):
        def rescale(vector):
            max = np.max(vector)
            min = np.min(vector)
            return (vector-min)/(max-min)
        return  np.apply_along_axis(rescale,0, ret)
if __name__ == "__main__":
    point = [0.1,0.5]
    subsamples = utils.subSampling(point,None,10)
    print(subsamples)
    plt.plot(point[0],'ro')
    plt.plot(point[1], 'bo')
    plt.scatter(np.zeros(10),subsamples[:,0],c = 'y')
    plt.scatter(np.zeros(10),subsamples[:,1],c = 'g')
    plt.show()