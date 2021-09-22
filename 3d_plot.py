#plots 3D plot from a .csv file
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

#mpl.rcParams['legend.fontsize'] = 10

df=pd.read_csv('philip_250.csv')
bmal=df['bmal_mRNA']
cry=df['cry_mRNA']
nrld=df['nrld_mRNA']
t=df['time']


fig = plt.figure()
ax = fig.gca(projection='3d')
z = bmal
x = nrld
y = cry
ax.plot(x, y, z, label='avg trajectory')
ax.set_xlabel('nrld')
ax.set_ylabel('cry')
ax.set_zlabel('bmal')
ax.legend()
