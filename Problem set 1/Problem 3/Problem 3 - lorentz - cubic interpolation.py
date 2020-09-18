import numpy as np
from matplotlib import pyplot as plt 
from scipy import interpolate

def lorentz(x):
    return 1/(1+x**2)

x = np.linspace(-1,1,7)
y = lorentz(x)

x_cubic = np.linspace(x[1],x[-2],1001)
y_true  = lorentz(x_cubic)
y_cubic = np.zeros(len(x_cubic))
for i in range(len(x_cubic)):    
    ind=np.max(np.where(x_cubic[i]>=x)[0])
    x_use=x[ind-1:ind+3]
    y_use=y[ind-1:ind+3]
    pars=np.polyfit(x_use,y_use,3)
    pred=np.polyval(pars,x_cubic[i])
    y_cubic[i]=pred

print('The rms error is ',np.std(y_cubic-y_true))

plt.plot(x,y,'*')
plt.plot(x_cubic,y_cubic)

plt.plot(x,y,'*', label='1/(1+x**2)')
plt.plot(x_cubic,y_cubic, label='Cubic Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.margins(0.1,0.1)
plt.grid()
plt.legend(loc="best")
plt.savefig('Problem_2_Lorentzian_CubicInter.pdf')
plt.show()