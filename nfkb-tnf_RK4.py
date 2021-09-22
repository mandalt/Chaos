

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sin, pi
import scipy.integrate as sp
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

#all rate constants have unit /min or uM/min 
init=[1.2,0.5,0,0,0]
init1=[0.1,0.5,0,0,0]
init2=[.005,0.5,0,0,0]
tmin=0
dt=0.05
iter_t=250000.
tmax=dt*iter_t
t=np.arange(tmin,tmax,dt)
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
A=0.24
v=1/60.

def nfkb(t,y):
    TNF=0.5 + A*sin(2*pi*v*t)
    N=y[0]
    Im=y[1]
    I=y[2]
    IKKa=y[3]
    IKKi=y[4]
    dNdt=kN_in*(N_tot-N)*(KI/(KI+I)) - kI_in*I*(N/(N+KN))
    dImdt=kt*(N**2) - gamma*Im
    dIdt=ktI*Im - alpha*IKKa*(N_tot-N)*(I/(I+KI))
    dIKKadt=ka*TNF*(ikk_tot - IKKa - IKKi) - ki*IKKa
    dIKKidt=ki*IKKa - kp*IKKi*(ka20/(ka20+a20*TNF))
    t=+dt
    x=np.asarray([dNdt,dImdt,dIdt,dIKKadt,dIKKidt])
    return x

soln=sp.solve_ivp(nfkb, t_span=(tmin,tmax),y0=init, method="RK45", t_eval=t,vectorized=True)
soln1=sp.solve_ivp(nfkb, t_span=(tmin,tmax),y0=init1, method="RK45", t_eval=t,vectorized=True)
soln2=sp.solve_ivp(nfkb, t_span=(tmin,tmax),y0=init2, method="RK45", t_eval=t,vectorized=True)
#print(soln)
time=soln.t
TNF=0.5 + A*np.sin(2*pi*v*time)

nucl_nfkb=5*soln.y[0]
cyto_ikb=soln.y[1]
ikb_m=soln.y[2]

nucl_nfkb1=5*soln1.y[0]
cyto_ikb1=soln1.y[1]
ikb_m1=soln1.y[2]

nucl_nfkb2=5*soln2.y[0]
cyto_ikb2=soln2.y[1]
ikb_m2=soln2.y[2]

print('done')


#mpl.rcParams['legend.fontsize'] = 10
#plots the NFkB oscillations driven by TNF
start=180000
#stop=iter_t
plt.plot(time[start:],nucl_nfkb[start:],"r", label="NFkB")
plt.plot(time[start:],TNF[start:],"g", label="tnf")
plt.figure()
plt.plot(time[start:],nucl_nfkb1[start:],"k", label="NFkB")
plt.plot(time[start:],TNF[start:],"g", label="tnf")
plt.figure()
plt.plot(time[start:],nucl_nfkb2[start:],"b", label="NFkB")
plt.plot(time[start:],TNF[start:],"g", label="tnf")
#plt.plot(t[20000:],c[20000:],"g", label="IkB/10")
#plt.plot(t[5000,:],)
#plt.xlabel("time(min)")
#plt.ylabel("NFkB or IkB")
#plt.title("Nuclear NFkB and cytoplasmic IkB oscillations without TNF")
plt.legend()
plt.show()

#plots the phase diagrame of the first 3 variable of the system. Shows the limit cycles and the chaotic attactor.
fig = plt.figure()
ax = fig.gca(projection='3d')
x = nucl_nfkb[start:]
y = cyto_ikb[start:]
z = ikb_m[start:]
ax.plot(x, y, z, linewidth=0.2,c='r', label='trajectory 1')

#fig = plt.figure()
ax = fig.gca(projection='3d')
x = nucl_nfkb1[start:]
y = cyto_ikb1[start:]
z = ikb_m1[start:]
ax.plot(x, y, z, c="k",linewidth=0.2, label='trajectory 2')

#fig = plt.figure()
ax = fig.gca(projection='3d')
x = nucl_nfkb2[start:]
y = cyto_ikb2[start:]
z = ikb_m2[start:]
ax.plot(x, y, z, c="b",linewidth=0.2 ,label='trajectory 3')
ax.set_ylabel('ikbm')
ax.set_zlabel('ikb')
ax.set_xlabel('nfkb')
ax.legend()
plt.show()





