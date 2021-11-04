
# Takes time serise data from a csv file and analyses it. Mostly finds frequency ratios and probabilities.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

timeSer = pd.read_csv('nfkb-noise002_10kh_5.csv')
#print(timeSer.head(5), timeSer.shape)
t = timeSer['time'].values
a = timeSer['initial1'].values
b = timeSer['initial2'].values
c = timeSer['initial3'].values
d = timeSer['initial4'].values
e = timeSer['initial5'].values

z = np.concatenate((a,b,c,d,e))
print(z.shape)

#finds the peaks and their occurance in time depending on the desired properties.

import scipy.signal as sig

peaks_a, _ = sig.find_peaks(a, 0.75, distance = 47)
peaks_b, _ = sig.find_peaks(b, 0.75, distance = 47)
peaks_c, _ = sig.find_peaks(c, 0.75, distance = 47)
peaks_d, _ = sig.find_peaks(d, 0.75, distance = 47)
peaks_e, _ = sig.find_peaks(e, 0.75, distance = 47)


peaka_loc = np.asarray(t[peaks_a])
peakb_loc = np.asarray(t[peaks_b])
peakc_loc = np.asarray(t[peaks_c])
peakd_loc = np.asarray(t[peaks_d])
peake_loc = np.asarray(t[peaks_e])


pos_a = (peaka_loc[1:]-peaka_loc[:-1])
pos_b = (peakb_loc[1:]-peakb_loc[:-1])
pos_c = (peakc_loc[1:]-peakc_loc[:-1])
pos_d = (peakd_loc[1:]-peakd_loc[:-1])
pos_e = (peake_loc[1:]-peake_loc[:-1])

#plots time period vs no. of oscillations. 
#This gives an estimate of how long the system spends in one mode of synchronsation.

freq_mode = np.concatenate((pos_a, pos_b, pos_c, pos_d, pos_e))
peak_loc = np.concatenate((peaka_loc, peakb_loc, peakc_loc, peakd_loc, peake_loc))
peaks = np.concatenate((peaks_a, peaks_b, peaks_c, peaks_d, peaks_e))

cycle_no = np.arange(0,len(freq_mode))

plt.plot(cycle_no[:-1], freq_mode[:-1], linewidth = 0.5)
plt.axhline(y = 100, linewidth = 1, c = 'r', label = '1:2')
plt.axhline(y = 150, linewidth = 1, c = 'g', label = '1:3')
plt.xlabel('oscillation number')
plt.ylabel('period(min)')
plt.legend()
plt.show()

#plots 2 trajectories for visualization purpose.
cycle_a = np.arange(1,len(peaks_a))
cycle_b = np.arange(1,len(peaks_b))

plt.plot(cycle_a, pos_a)
plt.show()
plt.plot(cycle_b, pos_b)
plt.show()


plt.figure()
#plt.scatter(peaka_loc[:-1],pos_a, s = 5, label='trajectory 1')
plt.plot(t,a, label='initial condition 1')
plt.figure()
#plt.scatter(peakb_loc[:-1],pos_b, s = 5, label='trajectory 2')
plt.plot(t,b, c='orange', label = 'initial condition 2')
plt.xlabel('time(min)')
plt.ylabel('frequency ratio')
plt.legend()

#plt.figure()
#plt.plot(t,a, label='initial condition 1')
#plt.xlabel('time(min)')
#plt.ylabel('NFkB')
#plt.legend()
#plt.figure()
#plt.plot(t,b, c='orange', label = 'initial condition 2')
#plt.xlabel('time(min)')
#plt.ylabel('NFkB')
#plt.legend()


# In[9]:


#plot a histogram of different states of synchronization.


lower = 2.05
upper = 2.95

freq_mode = np.concatenate((pos_a, pos_b))
plt.hist(freq_mode, density = True)

#print(len(pos))

sync2 = (np.count_nonzero((freq_mode < lower) | (freq_mode == lower)))/len(freq_mode)
transition = (np.count_nonzero((freq_mode > lower) & (freq_mode < upper)))/len(freq_mode)
sync3 = (np.count_nonzero((freq_mode > upper) | (freq_mode == upper)))/len(freq_mode)
#print(sync2/len(pos_a), transition/len(pos_a),sync3/len(pos_a))

# creating the dataset
data = {'1:2':sync2, 'transition':transition, '1:3':sync3}
modes = list(data.keys())
counts = list(data.values())
  
plt.figure()
 
# creating the bar plot
plt.bar(modes, counts, color ='maroon',
        width = 0.4)
 
#plt.xlabel("synchronization")
plt.ylabel("probability")
plt.title("synchronization modes")
#plt.legend()
plt.title('considering 5% interval')
plt.show()


# In[10]:


#count the no. of oscillations in each mode-locked state and the transition from one to the other.
#It considers that if the 3rd/5th peak of every peak that is outside the 5% interval is in same synchronised state then the jump did not happen. 

#df = pd.DataFrame(freq_mode, columns = ['modes'])

def screen(f):
    mode2, mode3, trans23, trans32 = 0,0,0,0
    for i in range(len(f)-5):
        if f[i] < lower:
            mode2 += 1
        elif f[i] > upper:
            mode3 += 1
        elif (f[i-5] < lower) & (f[i] > lower) & (f[i+5] > upper):
            trans23 += 1
        elif(f[i-5] > upper) & (f[i] < upper) & (f[i+5] < lower):
            trans32 += 1
        else:
            i += 5
    out = np.asarray([mode2, mode3, trans23, trans32])
    return out

mode_counts = np.apply_along_axis(screen, 0, freq_mode)

print(len(freq_mode),mode_counts, mode_counts.sum())

# creating the dataset
data = {'1:2':mode_counts[0], '2 to 3':mode_counts[2], '1:3':mode_counts[1], '3 to 2': mode_counts[3]}
modes = list(data.keys())
counts = list(data.values())
  
#fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(modes, counts, color ='green',
        width = 0.4)
 
#plt.xlabel("synchronization")
plt.ylabel("counts")
plt.title("synchronization modes")
#plt.legend()
plt.title('considering 5% interval')
plt.show()


# In[11]:


#count the no. of oscillations in each mode-locked state and the transition from one to the other.
#It considers that if the 5th peak of every peak that is outside the 5% interval is in same synchronised state then the jump did not happen. 
import numpy as np
lower = 2.05
upper = 2.95


def screen(f):
    mode2, mode3, trans23, trans32 = 0,0,0,0
    for i in range(len(f)-3):
        if f[i] < lower:
            mode2 += 1
        elif f[i] > upper:
            mode3 += 1
        elif (f[i-3] < lower) & (f[i] > lower) & (f[i+3] > upper):
            trans23 += 1
        elif (f[i-3] > upper) & (f[i] < upper) & (f[i+3] < lower):
            trans32 += 1
        else:
            i += 3
    out = np.asarray([mode2, mode3, trans23, trans32])
    return out

mode_counts = np.apply_along_axis(screen, 0, freq_mode)

print(len(freq_mode),mode_counts, mode_counts.sum())

# creating the dataset
data = {'1:2':mode_counts[0], '2 to 3':mode_counts[2], '1:3':mode_counts[1], '3 to 2': mode_counts[3]}
modes = list(data.keys())
counts = list(data.values())
  
#fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(modes, counts, color ='green',
        width = 0.4)
 
#plt.xlabel("synchronization")
plt.ylabel("counts")
plt.title("synchronization modes")
#plt.legend()
plt.title('considering 5% interval')
plt.show()


# In[132]:


#counts transitions from one synchronised mode to the other and their occurance in time.
    
def jump_loc(f,t):
    t_point = np.zeros(len(f)-3)
    jump_index = []
    for i in np.arange(0,len(f)-3,2):
        if (f[i-3] < lower) & (f[i] > lower) & (f[i+3] > upper):
            t_point[i] = f[i]
            jump_index.append(i)
        elif (f[i-3] > upper) & (f[i] < upper) & (f[i+3] < lower):
            t_point[i] = f[i]
            jump_index.append(i)
    return t_point, jump_index;

jump_a = jump_loc(pos_a, t)
jump_time_a = jump_a[0]
#a_jump = np.count_nonzero(jump_time_a)
jump_time_a[jump_time_a == 0] = np.nan
jump_b = jump_loc(pos_b, t)
jump_time_b = jump_b[0]
jump_time_b[jump_time_b == 0] = np.nan

jump_index = np.concatenate((jump_a[1], jump_b[1]))
print(len(jump_index))

plt.scatter(peaka_loc[:-4],jump_time_a, c='r',s = 10, label = 'transitions')
plt.plot(peaka_loc[:-1],pos_a, label='initial condition 1')
plt.xlabel('time')
plt.ylabel('frequency ratio')
plt.legend()

plt.figure()
plt.scatter(peakb_loc[:-4],jump_time_b, c='k', s = 10, label = 'transitions')
plt.plot(peakb_loc[:-1],pos_b, c = 'g',label='initial condition 2')
plt.xlabel('time')
plt.ylabel('frequency ratio')
plt.legend()



# In[12]:


#counts transitions out of one synchronised mode and their occurance in time.
    
def jump_loc(f,t):
    t_point = np.zeros(len(f)-3)
    jump_index = []
    for i in np.arange(0,len(f)-3,3):
        if (f[i] < lower) & (f[i+3] > lower):
            t_point[i] = f[i]
            jump_index.append(i)
        elif (f[i] > upper) & (f[i+3] < upper):
            t_point[i] = f[i]
            jump_index.append(i)
    return t_point, jump_index;

jump_a = jump_loc(pos_a, t)
jump_time_a = jump_a[0]
#a_jump = np.count_nonzero(jump_time_a)
jump_time_a[jump_time_a == 0] = np.nan
jump_b = jump_loc(pos_b, t)
jump_time_b = jump_b[0]
jump_time_b[jump_time_b == 0] = np.nan

jump_index = np.concatenate((jump_a[1], jump_b[1]))
print(len(jump_index))

plt.scatter(peaka_loc[:-4],jump_time_a, c='r',s = 10, label = 'transitions')
plt.plot(peaka_loc[:-1],pos_a, label='initial condition 1')
plt.xlabel('time')
plt.ylabel('frequency ratio')
plt.legend()

plt.figure()
plt.scatter(peakb_loc[:-4],jump_time_b, c='k', s = 10, label = 'transitions')
plt.plot(peakb_loc[:-1],pos_b, c = 'g',label='initial condition 2')
plt.xlabel('time')
plt.ylabel('frequency ratio')
plt.legend()


# In[134]:


#jump out of 1:2 mode

def jump_loc(f,t):
    t_point = np.zeros(len(f)-3)
    jump_index = []
    for i in np.arange(0,len(f)-3,3):
        if (f[i] < lower) & (f[i+3] > lower):
            t_point[i] = f[i]
            jump_index.append(i)
    return t_point, jump_index;

jump_2a = jump_loc(pos_a, t)
jump_time_2a = jump_2a[0]
#a_jump = np.count_nonzero(jump_time_a)
jump_time_2a[jump_time_2a == 0] = np.nan
jump_2b = jump_loc(pos_b, t)
jump_time_2b = jump_2b[0]
jump_time_2b[jump_time_2b == 0] = np.nan

jump_index_2 = np.concatenate((jump_2a[1], jump_2b[1]))
print(len(jump_index_2))

plt.scatter(peaka_loc[:-4],jump_time_2a, c='r',s = 10, label = 'transitions')
plt.plot(peaka_loc[:-1],pos_a, label='initial condition 1')
plt.xlabel('time')
plt.ylabel('frequency ratio')
plt.title('jumps out of 2')
plt.legend()

plt.figure()
plt.scatter(peakb_loc[:-4],jump_time_2b, c='k', s = 10, label = 'transitions')
plt.plot(peakb_loc[:-1],pos_b, c = 'g',label='initial condition 2')
plt.xlabel('time')
plt.ylabel('frequency ratio')
plt.title('jumps out of 2')
plt.legend()


# In[13]:


#jump out of 1:3 mode

def jump_loc(f,t):
    t_point = np.zeros(len(f)-3)
    jump_index = []
    for i in np.arange(0,len(f)-3,3):
        if (f[i] > upper) & (f[i+3] < upper):
            t_point[i] = f[i]
            jump_index.append(i)
    return t_point, jump_index;

jump_3a = jump_loc(pos_a, t)
jump_time_3a = jump_3a[0]
#a_jump = np.count_nonzero(jump_time_a)
jump_time_3a[jump_time_3a == 0] = np.nan
jump_3b = jump_loc(pos_b, t)
jump_time_3b = jump_3b[0]
jump_time_3b[jump_time_3b == 0] = np.nan

jump_index_3 = np.concatenate((jump_3a[1], jump_3b[1]))
print(len(jump_index_3))

plt.scatter(peaka_loc[:-4],jump_time_3a, c='r',s = 10, label = 'transitions')
plt.plot(peaka_loc[:-1],pos_a, label='initial condition 1')
plt.xlabel('time')
plt.ylabel('frequency ratio')
plt.title('jumps out of 3')
plt.legend()

plt.figure()
plt.scatter(peakb_loc[:-4],jump_time_3b, c='k', s = 10, label = 'transitions')
plt.plot(peakb_loc[:-1],pos_b, c = 'g',label='initial condition 2')
plt.xlabel('time')
plt.ylabel('frequency ratio')
plt.title('jumps out of 3')
plt.legend()


# In[18]:


#find the phase from time information
from math import pi

phase = np.zeros(1)
pi_phase = np.zeros(1)

for i in peaks:
    interval = np.linspace(i,(i+1),50)
    phase = np.append(phase, interval)
    circ = np.linspace(0,2*pi,50)
    pi_phase = np.append(pi_phase,circ)

jump_out = len(jump_index)
jump_phase = pi_phase[jump_index]
    
jump_out_2 = len(jump_index_2)
jump_phase_2 = pi_phase[jump_index_2]

jump_out_3 = len(jump_index_3)
jump_phase_3 = pi_phase[jump_index_3]

print('jump out of 1:2=',jump_out_2, 'jump out of 1:3=',jump_out_3)

plt.hist(jump_phase, bins = 10, density = True, color = 'grey', label = 'out of 1:2 or 1:3')
plt.xlabel('phase of the oscillator')
plt.ylabel('density')
plt.legend()

plt.figure()    
plt.hist(jump_phase_2, bins = 10, density = True, color = 'b', label = 'out of 2')
plt.xlabel('phase of the oscillator')
plt.ylabel('density')
plt.legend()
#plt.figure()
plt.hist(jump_phase_3, bins = 10, density = True, color = 'r', label = 'out of 3')
plt.xlabel('phase of the oscillator')
plt.ylabel('density')
plt.legend()


# In[15]:


phase_info = {'time':phase, 'jump phase':jump_phase}
phase_info = pd.DataFrame(phase_info, columns = ['time','phase','jump time'])
phase_info.to_csv('phase_info.csv')


# In[16]:


#create a bandpass

import scipy.signal as sig

lowcut = 0.07
order = 5

b1,b2 = sig.butter(order, lowcut, btype = 'low')
y = sig.filtfilt(b1,b2,a)

#print(y)
plt.plot(t,y)

#fft of the filtered signal

n=len(y)
f=np.fft.fft(y,n)
freq=np.fft.fftfreq(n)
freqp=freq[freq>=0]
psd=f * np.conj(f)/n
psd1=psd[0:len(freqp)]

plt.figure()
#plt.plot(freq,psd)
plt.yscale('log')
plt.plot(freqp,psd1, label='power density')
plt.xlabel('frequency')
plt.ylabel('psd')
plt.legend()


# In[17]:


#new_df=timeSer.loc[(timeSer['initial1'] > 0.9) & (timeSer['initial1'] < 1.)]
#print(new_df)

#plt.plot(timeSer['time'],timeSer['initial1'])
#plt.plot(timeSer['time'],timeSer['tnf'])
n=len(a)
f=np.fft.fft(a,n)
freq=np.fft.fftfreq(n)
freqp=freq[freq>=0]
psd=f * np.conj(f)/n
psd1=psd[0:len(freqp)]

#print(n)
plt.figure()
#plt.plot(freq,psd)
plt.yscale('log')
plt.plot(freqp,psd1, label='power density')
plt.xlabel('frequency')
plt.ylabel('psd')
plt.legend()


# In[ ]:




