import sys
import network
from data import loadData
import pandas as pd
import numpy as np
trainData, testData = loadData()
lr_inc_vec=np.arange(0.9,1.91,0.05)
lr_dec_vec=np.arange(0.1,1.11,0.05)
results = []
name=sys.argv[3]
name='inc_dec__'+name+'.csv'
for lr_inc in lr_inc_vec:
    for lr_dec in lr_dec_vec:
        result = [lr_inc,lr_dec]
        net = network.Network([13, int(sys.argv[1]), int(sys.argv[2]), 3])
        result.extend(net.train(trainData,testData,inc=lr_inc,dec=lr_dec))
        results.append(result)

results=pd.DataFrame(results)
results.to_csv(name,index=None,header=None)
