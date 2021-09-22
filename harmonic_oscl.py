#harmonic oscillator using RK4
import numpy as np
import matplotlib.pyplot as plt

tmin=0
tmax=20.
N=100.
dt=(tmax-tmin)/N
t=np.arange(tmin,tmax,dt)
#print(dt,t)
x=[0,5.]
q1=[]
p1=[]

def harosl(x,t):
    m=2.0
    sqw=.5
    q=x[0]
    p=x[1]
    y=np.zeros(len(x))
    y[0]= p/m
    y[1]= -m*sqw*q
    return y

def rk4(x,t,dt):
    for t in range(len(t)):
        k1= dt*harosl(x,t)
        k2= dt*harosl(x+k1*0.5,t+dt*0.5)
        k3= dt*harosl(x+k2*0.5,t+dt*0.5)
        k4= dt*harosl(x+k3, t+dt)
        z=x+(k1/6.0)+(k2/3.0)+(k3/3.0)+(k4/6.0)
        q1.append(x[0])
        p1.append(x[1])
        x=z
        
sol=rk4(x,t,dt)
#print(sol)
plt.scatter(t,q1)
plt.scatter(t,p1)
#plt.scatter(q1,p1)
