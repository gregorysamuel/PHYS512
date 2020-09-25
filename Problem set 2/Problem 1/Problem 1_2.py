import numpy as np 

def integrate_step(fun,x,tol):
    tol_i=tol
    count,ind,area,stop=0,0,0,0
    lvl_max=1
    fails=[]
    goal=x[-1]
    while stop!=1:
        lvl=tol_i/tol
        
        if (lvl_max<lvl) or(count==0):
            lvl_max=int(lvl)
            temp=x[ind]
            count+=1
            x=np.linspace(x[0],x[-1],4*(2**(lvl_max-1))+1)
            y=fun(x)
            ind=int(np.argwhere(x==temp)[0][0])

        step=int((np.argwhere(x==goal)[0][0]-ind)/4)
        area1=(x[ind+(4*step)]-x[ind])*(y[ind]+4*y[ind+2*step]+y[ind+4*step])/6
        area2=(x[ind+(4*step)]-x[ind])*( y[ind]+4*y[ind+1*step]+2*y[ind+2*step]+4*y[ind+3*step]+y[ind+4*step])/12
        test=np.abs(area1-area2)
        if test<tol and x[ind+4*step]!=x[-1]:
            print('Integrated from ',x[ind+0],' to ',x[ind+4*step])
            area=area+area2
            ind+=4*step
            tol*=2
            fails.remove(fails[-1])
            if len(fails)!=0:
                goal=fails[-1]
            else:
                goal=x[-1]
            continue
        elif test<tol and x[ind+4*step]==x[-1]:
            print('Integrated from ',x[ind+0],' to ',x[ind+4*step])
            area=area+area2
            break
        else:
            print('Couldnt integrate from ',x[ind+0],' to ',x[ind+4*step])
            fails.append(x[ind+2*step])
            tol=tol/2
            goal=fails[-1]
            continue
    
    return area,count

def lorentz(x):
    return 1/(1+x**2)

def sqr3(x):
    return 3*x**2

x=np.linspace(1,10,5)
ans,count=integrate_step(lorentz,x,0.001)
print(ans,count)