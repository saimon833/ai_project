import numpy as np
from csv import reader
from tabulate import tabulate
import network

def dataImport(name):
    with open(name, 'r', encoding='utf-8') as file:
        return [line for line in reader(file, delimiter=',')]

# Import data.tsv to dataList
dataList = dataImport('wine.csv')

def loadData(data):
    data = np.array(data)
    data=data.astype(float)
    mask = np.random.rand(len(data)) <= 0.8
    trainData = data[mask]
    testData = data[~mask]
    testIn, testOut = testData[:,1:], testData[:,:1]
    trainIn, trainOut = trainData[:,1:], trainData[:,:1]
    #przechowywać dane wyjściowe w postaci [x,y,z], gdzie zmienna która
    #ma być równa 1 odpowiadająca wyjściu a reszta 0
    testOutNew=list()
    for i in testOut:
        tmp=np.zeros(3)
        tmp[int(i)-1]=1
        testOutNew.append(tmp)
    testOut=np.array(testOutNew)
    testOut=testOut.astype(float)
    trainOutNew=list()
    for i in trainOut:
        tmp=np.zeros(3)
        tmp[int(i)-1]=1
        trainOutNew.append(tmp)
    trainOut=np.array(trainOutNew)
    trainOut=trainOut.astype(float)


    # Combining inputData and outputData in a single tuple
    trainData = [(np.array(trainIn[i], ndmin=2).T, np.array(trainOut[i], ndmin=2).T) for i in range(0, len(trainOut))]
    testData = [(np.array(testIn[i], ndmin=2).T, np.array(testOut[i], ndmin=2).T) for i in range(0, len(testOut))]
    return (trainData, testData)
# [attributes, hidden neurons, output]
net = network.Network([13,10,5, 3])
trainData, testData = loadData(dataList)
# (training_data, epochs, batch_size, eta, test_data)
net.SGD(trainData, 200, 10, 0.4, testData)
