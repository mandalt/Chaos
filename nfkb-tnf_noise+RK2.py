
# introduces noise to NFkB-TNF oscillations.
# solved using RK2.
# refer to 1991 paper by Honeycutt.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sin, pi, sqrt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

#all rate constants have unit /min or uM/min 
kN_in=5.4
kI_in=0.018
kt=1.03
ktI=0.24
KI=0.035
KN=0.029
gamma=0.017
alpha=1.05
N_tot=1.0
ka=0.24
ki=0.18
kp=0.036
ka20=0.0018
ikk_tot=2.0
a20=0.0026
init=[1.,.5,0,0,0]
init1=[.001,.5,0,0,0]
tmin=0
dt=0.005
iter_t=2000000
tmax=dt*iter_t
#time=np.arange(tmin,tmax,dt)
A=0.09
v=1/47.
output=np.zeros((4,iter_t))
q=0.005 #noise strength

def nfkb(x,t):
    N=x[0]
    Im=x[1]
    I=x[2]
    IKKa=x[3]
    IKKi=x[4]
    TNF=0.5 + A*sin(2*pi*v*t)
    y=np.zeros(len(x))
    y[0]=kN_in*(N_tot-N)*(KI/(KI+I)) - kI_in*I*(N/(N+KN))
    y[1]=kt*(N**2) - gamma*Im
    y[2]=ktI*Im - alpha*IKKa*(N_tot-N)*(I/(I+KI))
    y[3]=ka*TNF*(ikk_tot - IKKa - IKKi) - ki*IKKa
    y[4]=ki*IKKa - kp*IKKi*(ka20/(ka20+a20*TNF))
    return y

def dW(dt):
    """Sample a random number at each call."""
    r=np.random.normal(loc=0.0, scale=np.sqrt(1))
    dw=q*sqrt(dt)*r
    return dw

def rk2(x,dt):
    t=0
    nucl_N=[]
    ikbm=[]
    ikb=[]
    time=[]
    for i in range(iter_t):
        k1= dt*nfkb(x,t)
        k2= dt*nfkb(x+k1*dt+dW(dt),t)
        z=x+(k1+k2)/2.+ dW(dt)
        nucl_N.append(5*x[0])
        ikbm.append(x[1])
        ikb.append(x[2])
        x=z
        t=t+dt
        time.append(t)
    output[0]=time
    output[1]=nucl_N
    output[2]=ikbm
    output[3]=ikb
    return output
    
sol=rk2(init,dt)
output=np.zeros((4,iter_t))

sol1=rk2(init1,dt)
tnf_sig=0.5 + A*np.sin(2*pi*v*sol1[0,:])

print("done")   


#mpl.rcParams['legend.fontsize'] = 10
start=1420000
plt.plot(sol[0,start:],sol[1, start:],"b", label="5x NFkB")
plt.plot(sol[0,start:],tnf_sig[start:],"g", label="TNF signal")
plt.xlabel("time(min)")
plt.ylabel("NFkB or TNF")
plt.legend()
plt.show()
plt.figure()
plt.plot(sol1[0,start:],sol1[1,start:],"r", label="5x NFkB")
plt.plot(sol1[0,start:],tnf_sig[start:],"g", label="TNF signal")
plt.xlabel("time(min)")
plt.ylabel("NFkB or TNF")
#plt.title("Nuclear NFkB and cytoplasmic IkB oscillations without TNF")
plt.legend()
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
z = sol[3,start:]
x = sol[1,start:]
y = sol[2,start:]
ax.plot(x, y, z,linewidth=0.2, label='single trajectory')
ax.set_xlabel('nfkb')
ax.set_ylabel('ikb_mRNA')
ax.set_zlabel('ikb')
ax.legend()

#fig = plt.figure()
ax = fig.gca(projection='3d')
z = sol1[3,start:]
x = sol1[1,start:]
y = sol1[2,start:]
ax.plot(x, y, z, c='r',linewidth=0.2, label='single trajectory')
ax.set_xlabel('nfkb')
ax.set_ylabel('ikb_mRNA')
ax.set_zlabel('ikb_protein')
ax.legend()
plt.show()




