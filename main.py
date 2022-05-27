import network
from data import loadData
import pandas as pd
import numpy as np
from tabulate import tabulate
trainData, testData = loadData()
s1_vec = np.arange(1,15.01,1,dtype=int)
s2_vec = np.arange(1,15.01,1,dtype=int)
results = []
for s1 in s1_vec:
    for s2 in s2_vec:
        result = [s1,s2]
        net = network.Network([13, s1, s2, 3])
        print('S1: {0}, S2: {1}'.format(s1,s2))
        result.extend(net.SGD(trainData,1000,len(trainData),0.1,testData,0.0001,1.05,0.7))
        results.append(result)

results=pd.DataFrame(results)
results.to_csv('results_1.csv',index=None,header=None)
