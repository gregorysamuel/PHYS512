# -*- coding: utf-8 -*-
import numpy as np

def lorentz(x):
    return 1/(1+x**2)

def integrate_step(fun,x,tol_i):
    print('integrating from ',x[0],' to ',x[-1])
    tol=tol_i
    lvl=tol_i/tol
    if len(x):
        x=np.linspace(x[0],x[-1],4*lvl+1)
        y=fun(x)
        count+=1
    area1=(x[4]-x[0])*(y[0]+4*y[2]+y[4])/6
    area2=(x[4]-x[0])*( y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12
    myerr=np.abs(area1-area2)
    if myerr<tol:
        return area2,count
    else:   
        print('The ratio is ',myerr/tol)
        #xm=0.5*(x[0]+x[4])
        a1,count=integrate_step(fun,x,y,tol/2,ind)
        a2,count=integrate_step(fun,x,y,tol/2,count,tol_i,ind)
        return a1+a2,count

x=np.linspace(-10,10,5)
ans=integrate_step(lorentz,x,0.001,1)
