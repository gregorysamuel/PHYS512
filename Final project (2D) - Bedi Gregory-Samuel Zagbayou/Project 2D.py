import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import time

def greens(boundary,n,ndim=2):
    #get the potential from an unit of at (0,0)
    n=n+1-n%2 #decided to have odd number of elements, for central symetry
    dx=np.linspace(-boundary,boundary,n)
    shift=np.argwhere(dx**2==np.min(dx**2))[0][0]
    dx=np.roll(dx,-shift) #puting the 0 value at [0]
    if ndim==2:
        pot=np.zeros([n,n])
        xmat,ymat=np.meshgrid(dx,dx)
        dr=np.sqrt(xmat**2+ymat**2)
        dr[0,0]=1 #dial something in so we don't get errors
        pot=np.log(dr)
        pot[0,0]=pot[1,0]
        return pot

def white_noise2d(boundary,divs): #Gaussian noise in 2D
    return 2*boundary*np.random.randn(divs,divs)-boundary


class Particles:
    def __init__(self,x,v,m):
        self.x=x.copy()
        self.v=v.copy()
        try:
            self.m=m.copy()
        except:
            self.m=m*np.ones(x.shape[0])
        self.f=np.empty(x.shape)
        self.n=self.x.shape[0]
        self.ind=np.empty(x.shape)
        
    def get_density(self,boundary,divs):
        divs2    = divs+1-divs%2
        lim      = np.linspace(-boundary,boundary,divs2+1)
        lims     = [] #boundaries for the bins
        for i in range((self.x).shape[1]): #in each direction, the boundaries are the same
            lims.append(lim)
        #This gives the mass present in each box
        H,edges, numb=stats.binned_statistic_dd(self.x,self.m,statistic='sum',bins=lims)
        density  = H /( (2*boundary/divs2)**2 ) #mass divided by the area of a cell
        return density, edges
    
    
    def box_ind(self,edges): #indexes of the particles on the grid #
        self.ind[:,0]=np.digitize(self.x[:,0],edges[0],right='True')-1
        self.ind[:,1]=np.digitize(self.x[:,1],edges[1],right='True')-1
        
    def pkn(self,boundary,divs,d): #Gives the P=k^-n, power spectrum in 2D (Fourier space)
        k=np.linspace(-divs/2,divs/2,divs)*(2*np.pi/divs)
        shift=np.argwhere(k**2==np.min(k**2))[0][0]
        k=np.roll(k,-shift)
        ka,kb=np.meshgrid(k,k)
        dk=np.sqrt(ka**2+kb**2)
        dk[0,0]=1
        Ps=abs(dk**(-d))
        Ps[0,0]=0
        return Ps
    
    def test_real(self,boundary,divs,d):
        w_n   = white_noise2d(boundary, divs)
        w_nft = np.fft.fft2(w_n).real
        ps    = self.pkn(boundary,divs,d)
        
        #The power spectrum and white noise are now convolved
        # to obtain the mass repartition
        real2d=np.fft.irfft2(ps*w_nft,ps.shape)
        real2d=real2d-real2d.min()
        real2d=real2d/real2d.sum()
        
        #Assigning a random value to each particle
        #If said value is greater than the mass repartition at that point : Deletion
        vect_test=np.random.rand(self.n)
        density,edges = self.get_density(boundary, divs)
        self.box_ind(edges)
        indices=self.ind
        ax=indices[:,0].astype(int)
        ay=indices[:,1].astype(int)
        init_n=self.n
        del_m=np.argwhere(vect_test>real2d[ax%real2d.shape[0],ay%real2d.shape[1]])
        self.x=np.delete(self.x,del_m,0)
        self.v=np.delete(self.v,del_m,0)
        self.f=np.delete(self.f,del_m,0)
        self.m=np.delete(self.m,del_m,0)
        self.ind=np.delete(self.ind,del_m,0)
        self.n=self.x.shape[0]
        print('Accecptance rate',self.n/init_n)
        
            
    def get_forces(self,boundary,divs,mode='periodic'):
        if mode=='periodic':
            #We get the green function for one particle, centered on 0,0
            pot1         = greens(boundary,divs)
            #We get the density distribution over the grid
            density,edges= self.get_density(boundary,divs)
            pot          = np.fft.irfftn( np.fft.rfftn(density)*np.fft.rfftn(pot1), density.shape)
        else: #if the mode is not periodic, the potential is obtained with padding \
            #of the Green function and the density
            pot1           = greens(boundary,divs)
            pot1_pad       = np.pad(pot1,((0,pot1.shape[0]),(0,pot1.shape[1])),mode='constant')
            density, edges = self.get_density(boundary,divs)
            density_pad    = np.pad(density,((0,density.shape[0]),(0,density.shape[1])),mode='constant')
            pot            = np.fft.irfftn( np.fft.rfftn(density_pad)* \
                                           np.fft.rfftn(pot1_pad), density_pad.shape)
            newpot         = pot[pot.shape[0]//4:-pot.shape[0]//4,pot.shape[0]//4:-pot.shape[0]//4]
            pot            = np.fft.fftshift(newpot)
        
        #The length of each edge of the cube
        dx       = abs(edges[0][1]-edges[0][0])
        dy       = abs(edges[1][1]-edges[1][0])
        #The mass contained in each box of the grid
        mass_grid=density*(dx*dy)
        self.box_ind(edges) #indices of particles on the grid
        indices=self.ind.astype(int)
        
        #Finite differentiating the potential, with the centered approach
        field1= -(np.roll(pot,-1,axis=0)-np.roll(pot,1,axis=0))/(2*dx)
        field2= -(np.roll(pot,-1,axis=1)-np.roll(pot,1,axis=1))/(2*dy)
        
        #Approximation if the field is much smaller than the error on the derivative
        field1[abs(field1)<dx**3]=0
        field2[abs(field2)<dx**3]=0
        
        #Indexing made the position of the particles on the grid
        #mass_particle/mass_cell_grid is the contribution of the box to the force, weighted
        #multiplying this weight by the mass of the particle to obtain the force
        self.f[:,0]=field1[indices[:,0],indices[:,1]]*((self.m)**2)/mass_grid[indices[:,0],indices[:,1]]
        self.f[:,1]=field2[indices[:,0],indices[:,1]]*((self.m)**2)/mass_grid[indices[:,0],indices[:,1]]
        return pot,dx
            
        
    def update(self,dt,boundary,divs, mode='periodic'):
        
        #Leapfrog drift
        pot,dx = self.get_forces(boundary,divs,mode)
        gainx=dt*self.v
        if mode=='periodic':
            temp  = self.x+gainx
            self.x = (temp+boundary)%(2*boundary)-boundary
        else:
            self.x= self.x+gainx


        #Leapfrog kick
        a=self.f.copy()
        a[:,0]=a[:,0]/self.m
        a[:,1]=a[:,1]/self.m
        gainv=dt*a
        old =self.v.copy()
        self.v=old+gainv
        
        #Half of mv**2
        Ekin=((self.v[:,0]**2+self.v[:,1]**2)*self.m/2).sum()
        indices=self.ind.astype(int)
        #potential at the particles' position on the grid
        Epot=pot[indices[:,0],indices[:,1]].sum()
        E   =Ekin+Epot

        if mode!='periodic': #If the mode is not periodic, we may delete some particles
            #Lines where the particle is out of bounds in x, for deletion
            supx=np.argwhere(abs(self.x[:,0])>boundary).astype(int)
            self.x=np.delete(self.x,supx,0)
            self.v=np.delete(self.v,supx,0)
            self.f=np.delete(self.f,supx,0)
            self.m=np.delete(self.m,supx,0)
            self.ind=np.delete(self.ind,supx,0)
            self.n=self.x.shape[0]
            #Lines where the particle is out of bounds in y, for deletion
            supy=np.argwhere(abs(self.x[:,0])>boundary).astype(int)
            self.x=np.delete(self.x,supy,0)
            self.v=np.delete(self.v,supy,0)
            self.f=np.delete(self.f,supy,0)
            self.m=np.delete(self.m,supy,0)
            self.ind=np.delete(self.ind,supy,0)
        return E
    
boundary = 1.0
divs     = 75 
dt=0.00000005 #Obtain with CFL condition

#Random initialisation within the box
# np.random.seed(2)
# num        = int(1e7) #Number of paricle
# v0_max   = 0 #Initial max speed
# x        = 2*boundary*np.random.rand(num,2)-boundary
# v        = v0_max*np.random.rand(num,2)-v0_max/2
# m        = np.random.rand(num) 

# Two particles, circular orbit 
x        = np.asarray([[0,0],[-0.5,0]])
v        = np.asarray([[0,0],[0,3.7e4]])
m        = np.asarray([1e6,1]) 

#One particle
# x        = np.asarray([[0,0]])
# v        = np.asarray([[0,0]])
# m        = np.asarray([1]) 

#Initialization
part     = Particles(x,v,m)


    #Only for k-3 realization -- 3 is the exponent for the powerspectrum
#part.test_real(boundary,divs,3) #Prints the accpetance rate

## Plotting for the different setups
## Here only the case n=2

# #Plot part 1 (One particle)
# plt.clf()
# # Leapfrog - 1st Kick
# pot,dx=part.get_forces(boundary, divs)
# for i in range(part.n):
#     gainv=0.5*dt*(part.f[i,:]/part.m[i])   
#     part.v[i,:]=part.v[i,:]+gainv
# Energ=[]
# for i in range(750):
#     E=part.update(dt,boundary,divs)
#     plt.plot(part.x[0,0],part.x[0,1],'r.', markersize=6)
#     plt.xlim(-boundary,boundary)
#     plt.ylim(-boundary,boundary)
    # plt.savefig('1 particle frame number '+str(i)+'.png')
#     Energ.append(E)
#     #plt.pause(0.01)

# plt.clf()
# plt.plot(E)


# #Plot 2
# Leapfrog - 1st Kick
pot,dx=part.get_forces(boundary, divs)
Energ=[]
t0=time.time()
t_ini=t0
ite=3600
for i in range(part.n):
    gainv=0.5*dt*(part.f[i,:]/part.m[i])   
    part.v[i,:]=part.v[i,:]+gainv
plt.figure(0)
for i in range(ite):
    E=part.update(dt,boundary,divs)
    plt.clf()
    plt.plot(part.x[0,0],part.x[0,1],'r.', markersize=1)
    plt.plot(part.x[1,0],part.x[1,1],'b.', markersize=1)
    plt.xlim(-boundary,boundary)
    plt.ylim(-boundary,boundary)
    # plt.pause(0.01)
    # plt.savefig('2 particles frame number '+str(i)+'.png')
    Energ.append(E)
    if (i%5)==0:
        t1=time.time()
        del_t=t1-t0
        t0=t1
        print(i)
        print('Time since last stop: ',del_t)
        print('Time since begining: ',t0-t_ini)
# plt.figure(1)
plt.clf()
tim=np.arange(ite)*dt
plt.plot(tim,np.asarray(Energ)/Energ[0])
plt.xlabel('Time [$u.a$]')
plt.ylabel('Current total energy over initial total energy')
plt.savefig('Energy divided by initial energy- 2 - particles .png')        


# # Plot 3 -- periodic -- Also used for k-3 realization
# plt.clf()
# # Leapfrog - 1st Kick
# grad,dx=part.get_forces(boundary, divs)
# Energ=[]
# # for i in range(part.n):
# #     gainv=0.5*dt*(part.f[i,:]/part.m[i])   
# #     part.v[i,:]=part.v[i,:]+gainv
# a=part.f.copy()
# a[:,0]=a[:,0]/part.m
# a[:,1]=a[:,1]/part.m
# gainv=dt*a
# old =part.v.copy()
# part.v=old+gainv

# t0=time.time()
# t_ini=t0
# ite=3600
# for i in range(ite):
#     plt.clf()
#     E=part.update(dt,boundary,divs)
#     plt.scatter(part.x[:,0],part.x[:,1],color='k', s=1)
#     plt.xlim(-boundary,boundary)
#     plt.ylim(-boundary,boundary)
#     # plt.pause(0.01)
#     Energ.append(E)
#     t1=time.time()
#     del_t=t1-t0
#     t0=t1
#     plt.savefig('100 000 particles frame number - k-3 '+str(i)+'.png')
#     if (i%5)==0:
#         print(i)
#         print('Time since last iteration: ',del_t)
#         print('Time since begining: ',t0-t_ini)
# plt.clf()
# plt.plot(np.asarray(Energ)/Energ[0])
# tim=np.arange(ite)
# plt.xlabel('Time [$u.a$]')
# plt.ylabel('Current total energy over initial total energy')
# plt.savefig('Energy divided by initial energy- k-3 - 100 000particles .png')        

# Plot 3 -- non periodic
# plt.clf()
# #Making sure the mass is distributed approprietly
# #part.test_real(boundary, divs, n)
# # Leapfrog - 1st Kick
# grad,dx=part.get_forces(boundary, divs,mode='np')
# Energ=[]
# # for i in range(part.n):
# #     gainv=0.5*dt*(part.f[i,:]/part.m[i])   
# #     part.v[i,:]=part.v[i,:]+gainv
# a=part.f.copy()
# a[:,0]=a[:,0]/part.m
# a[:,1]=a[:,1]/part.m
# gainv=dt*a
# old =part.v.copy()
# part.v=old+gainv

# t0=time.time()
# t_ini=t0
# for i in range(50):
#     plt.clf()
#     E=part.update(dt,boundary,divs,mode='np')
#     plt.scatter(part.x[:,0],part.x[:,1],color='k', s=1)
#     plt.xlim(-boundary,boundary)
#     plt.ylim(-boundary,boundary)
#     plt.pause(0.01)
#     Energ.append(E)
#     t1=time.time()
#     del_t=t1-t0
#     t0=t1
#     # plt.savefig('100 000 particles frame number - periodic '+str(i)+'.png')
#     if (i%5)==0:
#         print(i)
#         print('Time since last iteration: ',del_t)
#         print('Time since begining: ',t0-t_ini)

















