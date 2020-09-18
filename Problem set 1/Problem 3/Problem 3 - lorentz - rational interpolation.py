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
    pars=np.dot(np.linalg.inv(mat),y) #changes from inv to pinv when the matrix becomes singular
    p=pars[:n]
    q=pars[n:]
    print(p)
    print(q)
    return p,q

def lorentz(x):
    return 1/(1+x**2)

x = np.linspace(-1,1,7)
y = lorentz(x)

n,m    = 3,5
p,q    = rat_fit(x,y,n,m)
x_rat  = np.linspace(x[0],x[-1],501)
y_true = lorentz(x_rat)
y_rat  = rat_eval(p,q,x_rat)

print('The rms error is ',np.std(y_rat-y_true))

plt.plot(x,y,'*', label='cos(x)')
plt.plot(x_rat,y_rat, label='Rational Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.margins(0.1,0.1)
plt.grid()
plt.legend(loc="best")
#plt.savefig('Problem_2_Lorentzian_Rational Interpolation with pinv instead of inv.pdf')
plt.show()