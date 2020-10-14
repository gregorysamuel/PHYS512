import numpy as np

x=np.loadtxt('dish_zenith.txt')
z,y,x=x[:,-1],x[:,1],x[:,0]
A=np.zeros((len(x),5))
A[:,0],A[:,1],A[:,2],A[:,3],A[:,4]=x**2,x,y**2,y,z
b=np.ones(len(x))

lhs=A.transpose()@A
rhs=A.transpose()@b
m=np.linalg.pinv(lhs)@rhs

u,s,v=np.linalg.svd(A,0)
fitp=v.transpose()@(np.diag(1/s)@(u.transpose()@b))
pred_svd=A@fitp

c1,c2,c3,c4,c5=fitp
a=-c1/c5
x0=c2/c5/a/2
y0=c4/c5/a/2
z0=(1/c5)-a*x0**2-a*y0**2

N=np.diag(A@m-b)**2
Ninv=np.linalg.pinv(N)

error_bars=np.sqrt(np.diag(np.linalg.pinv(A.transpose()@Ninv@A)))
uncertainty_a=a*((error_bars[0]/c1)**2 + (error_bars[-1]/c5)**2 )**0.5
f=1/4/a
uncertainty_f=f*((error_bars[0]/a)**2)**0.5