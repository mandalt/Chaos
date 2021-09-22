#logistic equation

def log():
    x0=np.random.rand(10)
    t=np.arange(20)
    out=np.zeros([len(x0),len(t)])
    out[:,0]=x0
    for j in range(1,len(t)):
        out[:,j]=2*out[:,j-1]*(1-out[:,j-1])
    return out

z=log()
t=np.arange(19)
plt.scatter(t,z[4,1:]) 



#logistic map bifurcation diagram

import numpy as np
import random
import matplotlib.pyplot as plt

x0=random.random()
t=np.arange(100)
r=np.arange(1,4,0.0005)
out=np.zeros((len(t+1),len(r)))
out[0]=x0

#print(r,x0,out)

for j in range(len(r)):
    for i in range(1,len(t)):
        out[i,j]=r[j]*out[i-1,j]*(1-out[i-1,j])

#print(out)
for k in range(90,100):
    plt.scatter(r,out[k],s=.1)
