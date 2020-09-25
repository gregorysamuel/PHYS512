import numpy as np
from matplotlib import pyplot as plt


def cheb_fit(fun,order):  #this function does a chebyshev fit and returns the coefficients for each polynomial
    x=np.linspace(-1,1,order+1)
    y=fun(x)
    mat=np.zeros([order+1,order+1])
    mat[:,0]=1
    mat[:,1]=x
    for i in range(1,order):
        mat[:,i+1]=2*x*mat[:,i]-mat[:,i-1]
    coeffs=np.linalg.pinv(mat)@y
    return coeffs

def leg_mat(rows,order):    #this functions returns the legendre polynomials (with coefficient 1) in a matrix
    mat = np.zeros([rows,order+1])
    mat[:,0]=1
    mat[:,1]=np.linspace(-1,1,rows)
    for i in range(1,order):
        mat[:,i+1]=( (2*i+1)*mat[:,1]*mat[:,i] - i*mat[:,i-1])/(i+1)
    return mat
    
def cheb_mat(rows,order): #this functions returns the chebyshev polynomials (with coefficient 1) in a matrix
    mat = np.zeros([rows,order+1])
    mat[:,0]=1
    mat[:,1]=np.linspace(-1,1,rows)
    for i in range(1,order):
        mat[:,i+1]= 2*mat[:,1]*mat[:,i] - mat[:,i-1]
    return mat

def newlog2(x): #defining the log in base 2 so that it maps the results of log 2 on [0.5,1] on [-1,1] instead
    return  np.log2(0.75 + x/4)
    

order=50 #choosing an arbitrary high order for a chebyshev fit
x=np.linspace(-1,1,order) 
cheb_coef=cheb_fit(newlog2,order) #fitting with chebyshev

trunc = 5 #the order at which we will truncate the chebyshev
rows  = 1000
leg_coef =np.polynomial.legendre.legfit(x,newlog2(x),trunc) #fitting to the same order( of the truncated chebyshev) with legendre
x     = np.linspace(-1,1,rows)
y     = newlog2(x) #the real values of log 2 on [0.5,1] mapped to [-1,1]
cheb  = cheb_mat(rows,trunc)@(cheb_coef[0:trunc+1]) #truncated chebyshev values of the fit
leg   = leg_mat(rows,trunc)@(leg_coef) #legendre values of the fit

res_cheb = y-cheb #residual of the truncated chebyshev fit
res_leg  = y-leg #residual of the legendre fit
real_x   = np.linspace(0.5,1,rows) 

print('The RMS error for the Chebyhev fit is',np.std(res_cheb)) #RMS error of the chebyshev fit
print('The RMS error for the Legendre fit is',np.std(res_leg))  #RMS error of the chebyshev fit
print('The max error for the Chebyhev fit is',np.max(abs(res_cheb))) #RMS error of the chebyshev fit
print('The max error for the Legendre fit is',np.max(abs(res_leg)))  #RMS error of the chebyshev fit
plt.plot(real_x,res_cheb,'*', label='Truncated Chebyshev')
plt.plot(real_x,res_leg,'--', label='Legendre')
plt.xlabel('x')
plt.ylabel('Residual')
plt.margins(0.1,0.1)
plt.grid()
plt.legend(loc="best")
plt.savefig('Problem_2.pdf')
plt.show()
