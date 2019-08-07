import numpy as np

def normalizeData(data):
    normData = np.array(data)
    for i in range(len(data[0])):
        maxVal = np.amax(data[:,i])
        minVal = np.amin(data[:,i])
        for j in range(len(data)):
            if maxVal != minVal:
                normData[j,i] = (normData[j,i] - minVal) / (maxVal - minVal)

    return normData