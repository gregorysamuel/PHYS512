import numpy as np
from matplotlib import pyplot as plt

#Convolution function
def conv(f,g):
    F=np.fft.fft(f)
    G=np.fft.fft(g)    
    return np.fft.ifft(F*G)

#Correlation function
def correl(f,g):
    a=np.fft.fft(f)
    b=np.conjugate(np.fft.fft(g))
    return np.fft.ifft(conv(a,b))
    
#Shifting function
def shift(f,d):
    F=np.fft.fft(f)
    temp=np.zeros_like(F)
    temp[d]=1    
    G=np.fft.fft(temp)    
    return np.fft.ifft(G*F)
    
N=128
x=np.linspace(-200,200,N+1)
tau=5

#Gaussian function
f=np.exp(-(x)**2/(2*tau**2))

#Will be useful to compute and store the maximum correlation at different shifts
d=range(64)
correlations=[]

#Computes the correlation at different shifts, stores the maximal vallue and plots every 8 operations
for i in d: 
    g=shift(f,i*N//len(d))
    correlation=correl(f,g)
    correlations.append(max(abs(correlation)))
    if (i%8)==0:    
        plt.plot(x,abs(correlation))
        plt.show()

#PLotting the max correlation against the shift on a semi log scale
plt.plot(np.asarray(d)*(N//len(d)),correlations)
plt.yscale('log')
plt.show()