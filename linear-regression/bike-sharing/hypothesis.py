import numpy as np

def hypothesis(x, theta):
    h = np.dot(x, theta)
    return h