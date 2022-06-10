import sys
import network
from data import loadData
import pandas as pd
import numpy as np
trainData, testData = loadData()
err_inc_vec=np.arange(1.01,1.091,0.01)
results = []
name=sys.argv[3]
name='err_inc'+name+'.csv'
for err in err_inc_vec:
    result = [err]
    net = network.Network([13, 7, 4, 3])
    result.extend(net.train(trainData,testData,inc=float(sys.argv[1]),dec=float(sys.argv[2]),err_inc=err))
    results.append(result)

results=pd.DataFrame(results)
results.to_csv(name,index=None,header=None)
