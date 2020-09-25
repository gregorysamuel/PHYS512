import numpy as np 

def integrate_step(fun,x,tol):
    tol_i=tol #fixing the initial value of the tolerence
    count,ind,area,stop=0,0,0,0 #setting the count, the index in the x matrix where we start calculating, the area calculating, and a stop value to kickstart the 'while'
    lvl_max=1 #the max level here is the maximum number of time the x axis has been divided by 2 (2 because its how we divide the tolerance when the integration fails)
    goals=[] #a list of the 'upper' values to where we will have to integrate 
    goal=x[-1] #the next upper value toward whic we integrate
    while stop!=1: #I used a while because I didnt understand correctly the first time and thought I should call the function while defining it
        lvl=tol_i/tol #the level is the number of times the x axis has been divided (in other words, its the relative tolerence)
        if (lvl_max<lvl) or(count==0): #if we go to a higher level than ever before (== lower tolerance ever)
            lvl_max=int(lvl)
            temp=x[ind]
            count+=1 #updating the count because we're about the call f(x)
            x=np.linspace(x[0],x[-1],4*(2**(lvl_max-1))+1) #we'll divide the set by 2 
            y=fun(x)
            ind=int(np.argwhere(x==temp)[0][0]) #and readjust the index to stay at the same value of x as before
        step=int((np.argwhere(x==goal)[0][0]-ind)/4) #we also readjust the step size according to the relative tolerance
        area1=(x[ind+(4*step)]-x[ind])*(y[ind]+4*y[ind+2*step]+y[ind+4*step])/6
        area2=(x[ind+(4*step)]-x[ind])*( y[ind]+4*y[ind+1*step]+2*y[ind+2*step]+4*y[ind+3*step]+y[ind+4*step])/12
        test=np.abs(area1-area2)
        if test<tol and x[ind+4*step]!=x[-1]: #if the tolerance is satisfied AND we're not at the upper limit
            print('Integrated from ',x[ind+0],' to ',x[ind+4*step]) #then we print the set upon which we integrated
            area=area+area2 #updating the area
            ind+=4*step #updating the index
            tol*=2 #updating the tolerance
            goals.remove(goals[-1]) #we update the goals list to remove the value that served as a upper 'goal' and that has been reached
            if len(goals)!=0: #if there are other elements in goals after the update
                goal=goals[-1] #then pick pick the previous element as the next goal for the integration
            else:
                goal=x[-1] #otherwise, we can aim for the overall upper bound
            continue #going straight to the next iteration
        elif test<tol and x[ind+4*step]==x[-1]: #if the tolerance is satisfied AND we're at the overall upper bound
            print('Integrated from ',x[ind+0],' to ',x[ind+4*step])
            area=area+area2 #final update on the area
            break #and we get out of the 'while'
        else: #if the tolerance condition is not satisfied
            print('Couldnt integrate from ',x[ind+0],' to ',x[ind+4*step])
            goals.append(x[ind+2*step]) #we update the list of upper bounds to which we will integrate
            tol=tol/2 #we slash the tolerance
            goal=goals[-1] #we update the current upper limit to reach
            continue    #going to the next iteration
    return area,count 

def lorentz(x):
    return 1/(1+x**2)

def sqr3(x):
    return 3*x**2

x=np.linspace(-np.pi,np.pi,5)
ans,count=integrate_step(np.sin,x,0.001)
print('The value is ',ans,'obtained with ',count,' calls to f(x)')