import numpy as np 
from scipy import integrate
from matplotlib import pyplot as plt 

def fun(u,R,z):
    return (z-R*u)/((R**2+z**2-2*R*z*u)**(3/2))

eps_0, mu_0=8.854e-12, 4*np.pi*1e-7
R,sig=1,1
u=np.linspace(-1,1,501)
const=(sig*R**2)/eps_0
z_0=np.linspace(0,10,501)
y_0=np.zeros_like(u)
y_1=y_0
h=2/len(u)
u_1= np.linspace(-1,1,1001)
h_1=2/len(u_1)
results=np.zeros_like(z_0)

for i in range(len(z_0)):
    f_0 = const*fun(u,R,z_0[i])
    f_1 = const*fun(u_1,R,z_0[i])
    results[i] = const*integrate.quad(fun,-1,1,(R,z_0[i]))[0]
    y_0[i] = 0.5*h*( f_0[0]+f_0[-1]+2*f_0[1:-1].sum() )
    y_1[i] = 0.5*h_1*( f_1[0]+f_1[-1]+2*f_1[1:-1].sum() )

plt.plot(np.delete(z_0,50),np.delete(results,50),'*',label='Quad')
plt.plot(z_0[50],results[50],'go', label='Quad at z=0')
romb = (y_0*h_1**2 - y_1*h**2)/(h_1**2-h**2) 
plt.plot(z_0,romb,'k',label='Romberg')
plt.plot(z_0[50],romb[50],'ro',label='Quad at z=0')
plt.xlabel('Distance from the center of the sphere')
plt.ylabel('Electric field')
plt.margins(0.1,0.1)
plt.grid()
plt.legend(loc="best")
print('For z=R, the quad integration gives ',results[50],' while the Romberg integration gives', romb[50])
plt.savefig('Problem_4.pdf')
plt.show()