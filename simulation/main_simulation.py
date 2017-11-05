import rpy2.robjects as robjects
r = robjects.r
from rpy2.robjects.numpy2ri import numpy2ri
numpy2ri.activate()
r.library("randomForest")
