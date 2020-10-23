import numpy as np
from matplotlib import pyplot as plt

#I make the assumption that we are working with real valued functions
def conv(f,g):
    length=max([len(f),len(g)])
    # zero padding at the end
    F,G= np.pad(f,[0,length],'constant'),np.pad(g,[0,length],'constant') 
    #FFT and shift so that the central frequency is the maximal frequency
    F,G=np.fft.fftshift(np.fft.fft(F)),np.fft.fftshift(np.fft.fft(G))    
    #IFT of the product to obtain the convolution in real space
    IFT=np.fft.ifft(F*G)
    if (length%2)==0:
        #slicing if the original length was even
        return  abs(IFT[(length//2):-(length//2)]) 
    else:
        #slicing if the original length was odd
        return  abs(IFT[(length//2):-(length//2)-1]) #we cut off the parts we don't need
    

N=100
x=np.linspace(-1,1,N+1)

#Gaussian fonction
tau=0.1
f=np.exp((-(x)**2)/(2*tau**2))

#Impulse fonction
g=np.zeros_like(x)
pos=15
g[pos]=1
print('Impulse at',x[pos])


#Plot of the convolution
convolution=conv(f,g)
plt.plot(x,convolution)