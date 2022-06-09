import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from data import __dataImport as dataImport
import pandas as pd

data = dataImport('S1_S2.csv')

data=pd.DataFrame(data)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#X=np.arange(15.01,1,-1,dtype=int)
X=np.arange(1,25.01,1,dtype=int)
Y=np.arange(1,25.01,1,dtype=int)
X,Y=np.meshgrid(X,Y)
Z = []
for i in range(25):
    tmp=[]
    for j in range(25):
        tmp.append(data[2][i*25+j])
    Z.append(tmp)
#print(Z)
Z=np.array(Z)
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)
ax.invert_zaxis()
ax.set_xlabel("S2")
ax.set_ylabel("S1")
ax.set_zlabel("% nauczenia sieci")
plt.show()
#plt.savefig("dupa.png")
