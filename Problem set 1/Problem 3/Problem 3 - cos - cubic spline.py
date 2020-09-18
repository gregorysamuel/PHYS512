import numpy as np
from matplotlib import pyplot as plt 
from scipy import interpolate


x = np.linspace(-np.pi/2,np.pi/2,15)
y = np.cos(x)

x_spline = np.linspace(x[0],x[-1],501)
cs       = interpolate.splrep(x,y)
y_spline = interpolate.splev(x_spline,cs)
print('The rms error is:', np.std(np.cos(x_spline)-y_spline))


plt.plot(x,y,'*', label='cos(x)')
plt.plot(x_spline,y_spline, label='Cubic Spline')
plt.xlabel('x')
plt.ylabel('y')
plt.margins(0.1,0.1)
plt.grid()
plt.legend(loc="best")
plt.savefig('Problem_2_cos_CubicSpline.pdf')
plt.show()