import numpy as np
from scipy.interpolate import splrep, splev 
from matplotlib import pyplot as plt

temperature, voltage, dvdt = np.loadtxt('lakeshore.txt')[:,0],np.loadtxt('lakeshore.txt')[:,1],np.loadtxt('lakeshore.txt')[:,2]
ordre= np.argsort(voltage) #taking the indices that would order the voltage from smallest to biggest
temperature, voltage, dvdt = temperature[ordre], voltage[ordre], dvdt[ordre] #ordering voltage, temperature and  
                                                                            #the derivative in ascending order of voltage
cs = splrep(voltage,temperature)
x  = np.linspace(voltage[0],voltage[-1],1000)
y  = splev(x, cs)
print(np.std(splev(voltage,cs)-temperature))

plt.plot(voltage, temperature,'*',label='Real values')
plt.plot(x,y,label='Cubic Spline')
plt.xlabel('Voltage')
plt.ylabel('Temperature')
plt.margins(0.1,0.1)
plt.grid()
plt.legend(loc="best")
plt.savefig('Problem_3.pdf')
plt.show()