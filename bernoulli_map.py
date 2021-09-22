#Bernoulli map

import numpy as np
import random
import matplotlib.pyplot as plt

def ber():
    x0=np.random.rand(1000)
    t=5
    out=np.zeros([len(x0),t])
    out[:,0]=x0
    for j in range(1,t):
        out[:,j]=2*out[:,j-1]
        for i in range(len(x0)):
            if out[i,j]>=1:
                out[i,j]=out[i,j]-1
            else:
                out[i,j]=out[i,j]
        
    return out
    

z=ber()
#print(z)
plt.scatter(z[:,0],z[:,4])

