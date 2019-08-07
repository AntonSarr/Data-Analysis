from hypothesis import hypothesis
from costFunc import costFunc
import numpy as np

def gradDesc(X, Y, theta, alpha, iterations):
    m = len(X)
    n = len(theta)
    temp = np.zeros((n,1))
    J_history = list()

    for i in range(iterations):
        temp = np.zeros((n,1))
        for j in range(n):
            for x, y in zip(X,Y):
                temp[j] += (alpha/m)*(hypothesis(x, theta) - y)*x[j]

        for j in range(n):
            theta[j] -= temp[j]

        J_history.append(costFunc(X,Y,theta))

    return theta, J_history