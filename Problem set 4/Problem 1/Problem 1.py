import numpy as np
from matplotlib import pyplot as plt

def shift(f,d):
    #FFT of our data
    F=np.fft.fft(f)
    
    #Impulse function, labelled temp, and its FFT
    temp=np.zeros_like(F)
    temp[d]=1    
    G=np.fft.fft(temp)    
    
    #Convolution of the two, which results in a shift 
    return np.fft.ifft(G*F)
    
N=100
x=np.linspace(-30,30,N+1)
tau=5
f=np.exp(-(x)**2/(2*tau**2))
d=N//4

#SHowing the shift
print('Shift of ',x[-1]+x[d])
plt.plot(x,abs(shift(f,d)))
plt.plot(x,f,'--')

