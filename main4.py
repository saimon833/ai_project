import network
from data import loadData
import pandas as pd
import numpy as np
from tabulate import tabulate
trainData, testData = loadData()
results = []
n=len(trainData)
size_vec = [n//x for x in range(1,50)]
for size in size_vec:
    result = [size]
    net = network.Network([13, 7, 4, 3])
    result.extend(net.train(trainData,testData,inc=1.55,dec=0.15,err_inc=1.04,mini_batch_size=size))
    results.append(result)

results=pd.DataFrame(results)
results.to_csv('batch.csv',index=None,header=None)
