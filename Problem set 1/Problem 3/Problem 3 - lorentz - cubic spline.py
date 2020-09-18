import numpy as np
from matplotlib import pyplot as plt 
from scipy import interpolate

def lorentz(x):
    return 1/(1+x**2)

x = np.linspace(-1,1,7)
y = lorentz(x)

x_spline = np.linspace(x[0],x[-1],501)
cs       = interpolate.splrep(x,y)
y_spline = interpolate.splev(x_spline,cs)
print('The rms error is:', np.std(lorentz(x_spline)-y_spline))


plt.plot(x,y,'*', label='cos(x)')
plt.plot(x_spline,y_spline, label='Cubic Spline')
plt.xlabel('x')
plt.ylabel('y')
plt.margins(0.1,0.1)
plt.grid()
plt.legend(loc="best")
plt.savefig('Problem_2_Lorentz_CubicSpline.pdf')
plt.show()