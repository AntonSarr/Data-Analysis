import numpy as np

def costFun(theta, Xold, Y):
    J = 0

    X = list(Xold)
    m = len(X)

    for i in range(m):
        X.insert(i, list([1, X.pop(i)]))

    n = len(X[0])

    for i in range(m):
        hypothesis = 0
        for j in range(n):
            hypothesis += X[i][j]*theta[j]
        J += 1 / (2 * m) * (hypothesis - Y[i]) * (hypothesis - Y[i])

    return J