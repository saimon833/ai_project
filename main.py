import numpy as np
from csv import reader
from tabulate import tabulate
import network

def dataImport(name):
    with open(name, 'r', encoding='utf-8') as file:
        return [line for line in reader(file, delimiter=',')]

# Import data.tsv to dataList
dataList = dataImport('wine.csv')

# Create numpy array from dataList
dataList = np.array(dataList)

inputData, outputData = dataList[:,1:], dataList[:,:1]
finalData = [(np.array(inputData[i], ndmin=2).T, np.array(outputData[i], ndmin=2).T) for i in range(0, len(outputData))]
print(finalData[0][0][0])


# [attributes, hidden neurons, output]
net = network.Network([13, 3, 1])

# (training_data, epochs, batch_size, eta, test_data)
net.SGD(finalData, 20, len(finalData), 0.9)
