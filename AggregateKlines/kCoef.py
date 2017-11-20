from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
import numpy as np
import bisect
from sklearn import datasets
from sklearn.model_selection import GridSearchCV

class KSparseLogisticRegression(object):
    
    @staticmethod
    def findBoundaryC(data, label, max_k = 5):
        time = 0
        if( max_k > len(data[0])):
            raise ValueError('max_k should be less than the number of features')

        found_min = False
        c_min = 1.0
        while not found_min:
            coef = LogisticRegression(C = c_min, penalty= 'l1', fit_intercept=True).fit(data,label).coef_[0]
            numNonzero = len(np.where(np.array(coef) != 0)[0])
            found_min = False if numNonzero > 0 else True
            c_min /= 10
            time += 1
            if(time >100): 
                raise RuntimeWarning('can not find the boundary value for min penalty c, current c = ' + str(c_min) )            
        found_max = False
        c_max = 1.0
        while not found_max:
            coef = LogisticRegression(C = c_max, penalty= 'l1', fit_intercept=True).fit(data,label).coef_[0]
            numNonzero = len(np.where(np.array(coef) != 0)[0])
            found_max = True if numNonzero >= max_k else False
            c_max *= 10
            time += 1
            if(time >100): 
                raise RuntimeWarning('can not find the boundary value for max penalty c, current c = ' + str(c_max))

        return [c_min,c_max]


    @staticmethod
    def findKsparseCoef(data,label,k):
        data = np.array(data)
        label = np.array(label)
        loop = 0
        found = False
        c_list_len = 10
        insert_loc = None
        c_limit = np.log(KSparseLogisticRegression.findBoundaryC(data,label,k))
        while not found:     
            if not insert_loc:
                C_list = np.logspace(c_limit[0],c_limit[1], num = c_list_len)
            else:
                start = np.log10(C_list[insert_loc-1]) 
                if insert_loc == len(C_list):
                    end = np.log10(C_list[-1])
                else:
                    end = np.log10(C_list[insert_loc])
                    
                C_list = np.logspace(start,end, num = c_list_len)
                
            clf = LogisticRegressionCV(Cs = C_list, penalty= 'l1',solver = 'liblinear')
            try:
                clf.fit(data,label)
                coef_path = clf.coefs_paths_[1][0]
            except:
                coef_path = []
                for c in C_list:                    
                    lr = LogisticRegression(penalty= 'l1',solver = 'liblinear', C = c)
                    lr.fit(data,label)

                    coef_path.append( lr.coef_[0].tolist()+lr.intercept_.tolist())



            def countNonzeros(coef_matrix):
                numberOfNonZeros = []
                for arr in coef_matrix:
                    # if the fit_intercept is on, remove the last element
                    arr = arr[:-1]
                    numberOfNonZeros.append(sum(np.array(arr) != 0))
                return numberOfNonZeros

            NonzerosList =  countNonzeros(coef_path)

            if k not in NonzerosList:
                
                insert_loc= bisect.bisect_left(NonzerosList,k)
                
                if loop > 10:
                    raise RuntimeError('can not find specific k, c_list_len = ' + str(c_list_len))
            else:
                found = True
                C_index = np.where(np.array(NonzerosList)==k)[0][0]


            loop += 1
        # ignore the last element, which represent the intercept
        NonzeroIndex = np.where(np.array(coef_path[C_index][:-1]) != 0)
        select_data = data[:,NonzeroIndex[0]]
        clf = LogisticRegression().fit(select_data,label)
        ret = np.zeros(data.shape[1]+1)
        ret[NonzeroIndex] = clf.coef_
        ret[-1] = clf.intercept_

        return ret



if __name__ == '__main__':
    

    digit_data,digit_label = datasets.load_digits(n_class = 2, return_X_y = True)
    # the intercept is appended as the last element
    k_coef = KSparseLogisticRegression.findKsparseCoef(digit_data,digit_label, 5)

    print(k_coef)