import pandas as pd
import numpy as np
from csv import reader
from tabulate import tabulate

def __genData():
    par_names = ['Class',
        'Alcohol',
     	'Malic acid',
     	'Ash',
    	'Alcalinity of ash',
     	'Magnesium',
    	'Total phenols',
     	'Flavanoids',
     	'Nonflavanoid phenols',
     	'Proanthocyanins',
    	'Color intensity',
     	'Hue',
     	'OD280/OD315 of diluted wines',
     	'Proline']
    
    data_list = pd.read_csv('./wine.data', sep=',', encoding='utf-8', header=None)#,names=par_names)
    
    
    # Copy of initial DataFrame
    norm_data = data_list.copy()
    # apply normalization techniques
    for column in norm_data.columns:
        if column !=0:
            norm_data[column] = (norm_data[column] - norm_data[column].min()) / (norm_data[column].max() - norm_data[column].min())
    
    norm_data.to_csv('wine.csv',header=False, index=False)
def __dataImport(name):
    with open(name, 'r', encoding='utf-8') as file:
        return [line for line in reader(file, delimiter=',')]

def loadData():
    data = __dataImport('wine.csv')
    data = np.array(data)
    data=data.astype(float)
    mask = np.random.rand(len(data)) <= 0.8
    trainData = data[mask]
    testData = data[~mask]
    testIn, testOut = testData[:,1:], testData[:,:1]
    trainIn, trainOut = trainData[:,1:], trainData[:,:1]
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


    trainData = [(np.array(trainIn[i], ndmin=2).T, np.array(trainOut[i], ndmin=2).T) for i in range(0, len(trainOut))]
    testData = [(np.array(testIn[i], ndmin=2).T, np.array(testOut[i], ndmin=2).T) for i in range(0, len(testOut))]
    return (trainData, testData)

if __name__ == '__main__':
    __genData()
