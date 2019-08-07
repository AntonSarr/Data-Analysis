import pandas as pnd
import numpy as np
from normalizeData import normalizeData
import matplotlib.pyplot as plt
from gradDesc import gradDesc

pandasData = pnd.read_csv("data.txt", names=['Размер дома, фут^2', 'Количество спален', 'Стоимость дома, $'], skip_blank_lines=True)
numpyData = np.zeros((len(pandasData.values), len(pandasData.values[0]) + 1))
numpyData[:,1:] = pandasData.values
numpyData[:,0] = 1

print(pandasData)

normData = normalizeData(numpyData)

theta = np.zeros((3,1))
iterations = 1500
alpha = 0.1

J_history = list()
theta, J_history = gradDesc(normData[:, 0:-1], normData[:,-1], theta, alpha, iterations)

plt.plot(range(iterations),J_history)
plt.show()