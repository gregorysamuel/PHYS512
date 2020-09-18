import numpy as np
from matplotlib import pyplot as plt 
from scipy import interpolate

def rat_eval(p,q,x):
    top=0
    for i in range(len(p)):
        top=top+p[i]*x**i
    bot=1
    for i in range(len(q)):
        bot=bot+q[i]*x**(i+1)
    return top/bot

def rat_fit(x,y,n,m):
    assert(len(x)==n+m-1)
    assert(len(y)==len(x))
    mat=np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=x**i
    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i
    pars=np.dot(np.linalg.inv(mat),y)
    p=pars[:n]
    q=pars[n:]
    return p,q


x = np.linspace(-np.pi/2,np.pi/2,15)
y = np.cos(x)

n,m    = 7,9
p,q    = rat_fit(x,y,n,m)
x_rat  = np.linspace(x[0],x[-1],501)
y_true = np.cos(x_rat)
y_rat  = rat_eval(p,q,x_rat)

print('The rms error is ',np.std(y_rat-y_true))


plt.plot(x,y,'*', label='cos(x)')
plt.plot(x_rat,y_rat, label='Rational Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.margins(0.1,0.1)
plt.grid()
plt.legend(loc="best")
plt.savefig('Problem_2_cos_Rational Interpolation.pdf')
plt.show()