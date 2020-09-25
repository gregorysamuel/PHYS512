import numpy as np
from matplotlib import pyplot as plt

def get_chev_matrix(x0,x1,order):
    mat=np.zeros([order+1,order+1])
    mat[:,0]=1
    if order<1:
        return mat
    elif order==1:
        mat[:,1]=np.linspace(x0,x1,order+1)
        return 
    else:
        mat[:,1]=np.linspace(x0,x1,order+1)
        for i in range(1,order):
            mat[:,i+1]=2*mat[:,1]*mat[:,i]-mat[:,i-1]      
        return mat

#def cheb_coeffs(fun,order):
#    coeffs=np.zeros(order)
#    for i in range(order):
#        alpha=np.zeros(order)
#        for j in range(order):
#            alpha[j]=fun( np.cos( np.pi*(j+0.5)/order) )*np.cos( i*np.pi*(j+0.5)/order)
#        coeffs[i]= (2/order)*np.sum(alpha)
#    return coeffs

def cheb_coeffs(fun,order,A,x):
    coeffs=np.zeros(order)
    for i in range(order):
        alpha=np.zeros(order)
        for j in range(order):
            alpha[j]=fun(x[j])*A[:,i][j]
        coeffs[i]= (2/order)*np.sum(alpha)
    return coeffs

tol=1e-6
err=1
order=10
while err>tol:
    true_x= np.linspace(0.5,1,order+1)
    true_y= np.log2(true_x)
    #new_x=(true_x-0.5*(1+0.5))/(0.5*(1-0.5))
    A=get_chev_matrix(true_x[0],true_x[-1],order)
    coeffs=cheb_coeffs(np.log2,order+1,A,true_x)
    cheb_fit=A@coeffs-0.5*coeffs[0]
    err=np.std(true_y-cheb_fit)
    order+=1
    
plt.plot(true_x, true_y,'o')
plt.plot(true_x, cheb_fit,'x')