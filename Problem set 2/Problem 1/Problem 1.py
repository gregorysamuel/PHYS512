import numpy as np 
from matplotlib import pyplot as plt 

def integrate_step(fun,x_i,x_f,tol):
    test=tol+1
    count=0
    area=0
    xf=x_f
    arret=0
    while (test>tol) or (arret!=2):
        count+=1
        x=np.linspace(x_i,x_f,5)
        y=fun(x)
        area1=(x[-1]-x[0])*(y[0]+4*y[2]+y[4])/6
        area2=(x[-1]-x[0])*( y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12
        test=np.abs(area1-area2)
        if test<tol:
            print('It worked from ',x[0],' to ',x[-1])
            area=area+area2
            x_i=x[-1]
            x_f=x[-1]+2*(x[-1]-x[0])
            if x_f>xf:
                x_f=xf
                arret+=1
            tol*=2
            continue
        else:
            x_i=x[0]
            x_f=x[0]+(x[-1]-x[0])/2
            tol=tol/2
            continue
    
    return area,count

def lorentz(x):
    return 1/(1+x**2)

ans,count=integrate_step(lorentz,-10,10,0.001)
print(ans,count)