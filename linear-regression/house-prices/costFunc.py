from hypothesis import hypothesis

def costFunc(X, Y, theta):
    J = 0
    m = len(X)
    n = len(X[0])

    for row,y in zip(X,Y):
        J += 1/(2*m)*pow((hypothesis(row,theta) - y), 2)

    return J