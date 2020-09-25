import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate

half_life=[]        #List cointaining the half lives of the different elements, in the same order as in the slides
half_life.append(4.468e9*365.25*24*60*60)
half_life.append(24.10*24*60*60)
half_life.append(6.7*60*60)
half_life.append(245500*365.25*24*60*60)
half_life.append(75830*365.25*24*60*60)
half_life.append(1600*365.25*24*60*60)
half_life.append(3.8235*24*60*60)
half_life.append(3.1*60)
half_life.append(26.8*60)
half_life.append(19.9*60)
half_life.append(164.3e-6)
half_life.append(22.3*365.25*24*60*60)
half_life.append(5.015*365.25*24*60*60)
half_life.append(138.376*24*60*60)

def fun(t,N,half_life=half_life):       #the differential equations linking the number of atoms per elements in time
    dNdt=np.zeros(len(half_life)+1)
    dNdt[0]=-N[0]/half_life[0]
    for i in range(len(dNdt)-2):
        dNdt[i+1]=(N[i]/half_life[i]) - N[i+1]/half_life[i+1]
    dNdt[-1]=N[-2]/half_life[-2]
    return dNdt

N0=np.zeros(len(half_life)+1) #number of atoms at the beginning for each element. Set to 0 for all
N0[0]=1e23 #number of atoms of Uranium 238 at the beginning 
t0,t1=0,4.468e9*365.25*24*60*60 #time at the beginning, time at the end(chosen to fit the wanted timescale)
ans_stiff= integrate.solve_ivp(fun,[t0,t1],N0,method='Radau') #implicit solver for the stiff equation
ur238_evol=ans_stiff.y[0,:] #evolution of the number of Uranium 238 in time
ur234_evol=ans_stiff.y[3,:] #evolution of the number of Uranium 234 in time
th_evol=ans_stiff.y[4,:] #evolution of the number of Thorium 230 in time
pb_evol=ans_stiff.y[-1,:] #evolution of the number of Pb206 in time
time=ans_stiff.t #time steps

plt.plot(time[71:90],ur234_evol[71:90],'r',label='Ur234') #can be changed to see the evolution of another element
plt.plot(time[78:90],th_evol[78:90],'g',label='Th230') #can be changed to see the evolution of another element
plt.xlabel('Time')
plt.ylabel("Number of atoms")
plt.yscale('log')
plt.margins(0.1,0.1)
plt.grid()
plt.legend(loc="best")
plt.savefig('Problem_3_Ur234_to_Th230.pdf')
plt.show()


