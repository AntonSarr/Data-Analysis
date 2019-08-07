import numpy as np
from hypothesis import hypothesis

def costFunc(X, Y, theta):
    J = 0
    m = X.shape[0]
    for (x,y) in zip(X,Y):
        J += 1/(2*m)*pow((hypothesis(x, theta) - y),2)
    return J