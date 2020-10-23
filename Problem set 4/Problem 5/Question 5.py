import numpy as np
from matplotlib import pyplot as plt

def delta(k,alpha):
#    return 1*((k)==(np.ones_like(k)*alpha))
    if k==alpha:
        return 1
    else:
        return 0

N=128
frequence=4.1/N

#Numerical DFT
x=np.arange(0,N)
computed=np.sin(2*np.pi*frequence*x)
windowed=computed*(0.5-0.5*np.cos(2*np.pi*x/N))
DFT=np.fft.fft(computed)


#Windowed version
DFT_convolved=np.fft.fft(windowed)

#Plots
plt.plot(abs(DFT),'--')
plt.plot(abs(DFT_convolved),'--')
plt.yscale('log')

#Analytical DFT
k=-N*frequence
FFT=(N/(2j))*(delta(k,-frequence*N)-delta(k,N*frequence))
print('The total error on the analytical estimate is ',np.mean(abs(DFT))-2*abs(FFT)/128)