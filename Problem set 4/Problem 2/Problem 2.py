import numpy as np
from matplotlib import pyplot as plt

def conv(f,g):
    F=np.fft.fft(f)
    G=np.fft.fft(g)    
    return np.fft.ifft(F*G)

def correl(f,g):
    a=np.fft.fft(f)
    b=np.conjugate(np.fft.fft(g))
    return np.fft.ifft(conv(a,b))
    
N=100
x=np.linspace(-30,30,N+1)
tau=5
f=np.exp(-(x)**2/(2*tau**2))
cor=correl(f,f)
plt.plot(x,abs(cor))