import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate

half_life=[]
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

def fun(t,N,half_life=half_life):
    dNdt=np.zeros(len(half_life)+1)
    dNdt[0]=-N[0]/half_life[0]
    for i in range(len(dNdt)-2):
        dNdt[i+1]=(N[i]/half_life[i]) - N[i+1]/half_life[i+1]
    dNdt[-1]=N[-2]/half_life[-2]
    return dNdt

N0=np.zeros(len(half_life)+1)
N0[0]=1e23
t0,t1=0,1e6*365.25*24*60*60
ans_stiff= integrate.solve_ivp(fun,[t0,t1],N0,method='Radau')