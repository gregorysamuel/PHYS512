import numpy as np 
from matplotlib import pyplot as plt 

def dfdx(fun,x,delta): #this function evaluates df/dx using the formula I derived with f(x+-delta) and f(x+-2*delta) 
    return  (fun(x+delta)-fun(x-delta)+(fun(x-2*delta)-fun(x+2*delta))/4  ) / delta

x = np.linspace(-5,5,1001)
const = 1 #change this from 1 to 0.01 to go from exp(x) to exp(0.01x)
fun = np.exp 
delta=np.asarray([1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12]) #using an array of deltas to map errors against the deltas 
val=np.zeros_like(delta)
for i in range(len(delta)):
    dfundx = dfdx(fun,const*x,delta[i]) #evaluates df/dx with a given delta
    val[i] = np.std(fun(const*x)-dfundx) #stores the values of the mean error

plt.plot(delta,val,'*', clip_on='False')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('delta')
plt.ylabel('Standard deviation on the error')
plt.grid()
plt.savefig('Question_1_expx.pdf')
plt.show()

