import numpy as np
from matplotlib import pyplot as plt


def cheb_fit(fun,order):
    x=np.linspace(-1,1,order+1)
    y=fun(x)
    mat=np.zeros([order+1,order+1])
    mat[:,0]=1
    mat[:,1]=x
    for i in range(1,order):
        mat[:,i+1]=2*x*mat[:,i]-mat[:,i-1]
    coeffs=np.linalg.pinv(mat)@y
    return coeffs

def leg_mat(rows,order):
    mat = np.zeros([rows,order+1])
    mat[:,0]=1
    mat[:,1]=np.linspace(-1,1,rows)
    for i in range(1,order):
        mat[:,i+1]=( (2*i+1)*mat[:,1]*mat[:,i] - i*mat[:,i-1])/(i+1)
    return mat
    
def cheb_mat(rows,order):
    mat = np.zeros([rows,order+1])
    mat[:,0]=1
    mat[:,1]=np.linspace(-1,1,rows)
    for i in range(1,order):
        mat[:,i+1]= 2*mat[:,1]*mat[:,i] - mat[:,i-1]
    return mat

def newlog2(x):
    y=np.log2(0.75 + x/4)
    return  y #+ 0.01*np.random.randn(len(x))
    

order=50
x=np.linspace(-1,1,50)
cheb_coef=cheb_fit(newlog2,order)
leg_coef=np.polynomial.legendre.legfit(x,newlog2(x),order)

trunc = 5
rows  = 1000
x     = np.linspace(-1,1,rows)
y     = newlog2(x)
cheb  = cheb_mat(rows,trunc)@(cheb_coef[0:trunc+1])
leg   = leg_mat(rows,trunc)@(leg_coef[0:trunc+1])

res_cheb = y-cheb
res_leg  = y-leg

print(np.std(res_cheb))
plt.plot(x,cheb,'*')
plt.plot(x,leg,'--')
