import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt
from costFunc import costFunc
from gradDesc import gradDesc

pandasData = pnd.read_csv("day.csv", usecols=['workingday', 'atemp', 'weathersit', 'cnt'])

# Importing data
data = pandasData.values
X = data[:, :3]
XwithOnes = np.empty((X.shape[0], X.shape[1] + 1))
XwithOnes[:, 1:] = X
XwithOnes[:, 0] = 1
Y = data[:, 3]

# Taking a look at data
print("All data: ")
print(pandasData)

# Initializing theta (weights)
theta = np.zeros(XwithOnes.shape[1])

# Initializing alpha
alpha = 0.01

# Initialiaing number of iterations in gradient descent
iterations = 150

# List of cost functions values during implementing gradient descent
J_history = []

# Implementing gradient descent
theta, J_history = gradDesc(XwithOnes, Y, theta, alpha, iterations)

# Checking if cost function is decreasing
plt.plot(range(iterations), J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Value of the cost function')
plt.grid(linewidth=0.8, color='grey', alpha=0.3)
plt.show()

