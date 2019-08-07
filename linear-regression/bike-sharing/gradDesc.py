import numpy as np
from costFunc import costFunc
from hypothesis import hypothesis
from costFunc import costFunc

def gradDesc(X, Y, theta, alpha, iterations):
    J_history = list()
    m = X.shape[0]
    n = X.shape[1]

    for iter in range(iterations):

        temp = np.zeros(n)

        for j in range(n):
            for (x,y) in zip(X,Y):
                temp[j] += (alpha/m)*(hypothesis(x, theta) - y)*x[j]

        for j in range(n):
            theta[j] -= temp[j]

        J_history.append(costFunc(X,Y, theta))

    return theta, J_history