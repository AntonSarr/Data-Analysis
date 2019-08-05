import matplotlib.pyplot as plt
import numpy as np
import pandas as pn
from gradDesc import gradDesc
from costFun import costFun

x = []
y = []

dataSrc = open("ex1data1.txt")
dataTxt = dataSrc.read()
dataTxt = dataTxt.split("\n")

for i in range(len(dataTxt) - 1):
    dataTxt[i] = dataTxt[i].split(",")
    x.append(float(dataTxt[i][0]))
    y.append(float(dataTxt[i][1]))

dataPnd = pn.DataFrame(dataTxt, columns=['Размер города', 'Доход фудтрака'])
dataPnd = dataPnd.dropna()
print(dataPnd)

xNorm = list()
yNorm = list()
for i in range(len(x)):
    xNorm.append((x[i] - min(x))/(max(x) - min(x)))
    yNorm.append((y[i] - min(y))/(max(y) - min(x)))


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams["figure.figsize"] = [10, 5]
plt.xlabel("Размер города, 10.000 человек")
plt.ylabel("Доход фудтрака, $10.000")
plt.title("Зависимость дохода фудтрака от размера города, в котором он размещается")
plt.xlim([0.5, 30.5])
plt.grid(linewidth=0.8, color='grey', alpha=0.3)
plt.scatter(x,y)

m = len(x)
theta = [0]*2

iterations = 1500
alpha = 0.01
theta, J_history = gradDesc(theta, alpha, iterations, x, y)

x_hyp = np.linspace(0,35,100)
y_hyp = np.array(theta[0] + x_hyp*theta[1])

plt.plot(x_hyp,y_hyp)

plt.show()
plt.close()

plt.rcParams["figure.figsize"] = [10, 5]
plt.xlabel("Итерации")
plt.ylabel("Cost Function")
plt.title("Зависимость функции потерь от количества итераций градиентного спуска")
plt.xlim([0, iterations])
plt.grid(linewidth=0.8, color='grey', alpha=0.3)
plt.plot(range(iterations),J_history)
plt.show()