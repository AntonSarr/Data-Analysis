from costFun import costFun

def gradDesc(thetaOld, alpha, iter, Xold, Y):
    new_theta = []
    X = list(Xold)
    theta = list(thetaOld)
    for i in range(len(X)):
        X.insert(i,list([1,X.pop(i)]))
    m = len(X)
    n = len(X[0])

    J_history = list()

    for num_iters in range(iter):

        hypothesis = list()

        for i in range(m):
            temp = 0
            for j in range(n):
                temp += X[i][j]*theta[j]
            hypothesis.append(temp)

        for j in range(n):
            temp = 0
            for i in range(m):
                temp += (alpha/m)*(hypothesis[i] - Y[i])*X[i][j]
            theta[j] = theta[j] - temp

        J_history.append(costFun(theta, Xold, Y))

    return theta, J_history