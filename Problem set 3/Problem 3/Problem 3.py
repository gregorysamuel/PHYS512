import numpy as np
import camb
from matplotlib import pyplot as plt

def deriv(func,steps,pars,length): 
    good_indices=np.argwhere(steps!=0).transpose()[0,:] #the 'good indices' are those where we 
    #will compute the derivative
    der=np.zeros((length,len(good_indices))) 
    for i in range(len(good_indices)):    
        h=steps[good_indices[i]] #the step on the variable where we are computing the derivative
        parsm1,pars1=pars.copy(),pars.copy()
        parsm1[good_indices[i]]=pars[good_indices[i]]-h
        pars1[good_indices[i]]=pars[good_indices[i]]+h
        fm1=func(parsm1)
        f1=func(pars1)
        der[:,i]= (f1-fm1)/(2*h) #central value derivative
    return der[2:length+2] #making sure that its valid at the borders 


def get_spectrum(pars,lmax=2000):
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt

def lm_camb(pars_init,steps,func,y,Ninv): #levenberg-Marquardt algorithm
    good_indices=np.argwhere(steps!=0).transpose() #I use these as indices for the derivative 
    #if the steps[i] is 0, then I don't derivate at that position, and its not a 'good indice'
    pars= pars_init.copy()
    pred=func(pars_init)    
    y_pred=pred[2:len(y)+2] #cutting out the first two values, as l=0 and l=1 there
    lamb_cur=0    #initial lambda value
    derivs=deriv(func,steps,pars_init,len(y)) #matrix with the derivatives
    resid=y-y_pred #data minus current model
    chi_cur=resid.transpose()@Ninv@resid 
    rhs= derivs.transpose()@(Ninv@resid)
    lhs= derivs.transpose()@Ninv@derivs
    #print(chi_cur)
    for iter in range(10): #should have been a 'while', testing on the convergence
        #but I didn't change it, as it doesnt matter much in this case
        lhs                   = (derivs.transpose()@Ninv@derivs)+(lamb_cur*np.diag(np.diag(derivs.transpose()@Ninv@derivs)))
        rhs                   = derivs.transpose()@(Ninv@resid)    
        step                  = np.linalg.pinv(lhs)@rhs
        pars_new              = pars
        pars_new[good_indices]= pars[good_indices]+step #only affecting the indexes where we derivated
        y_pred_new            = func(pars_new)[2:len(y)+2]
        resid_new             = y-y_pred_new #data minus current model
        chi_new               = resid_new.transpose()@Ninv@resid_new
        var_chi=chi_new-chi_cur
        if chi_new>=chi_cur and lamb_cur==0: 
            lamb_cur=1 #lambda becomes 1 if it was 0 before and chi didn't improve
        elif chi_new>chi_cur:
            lamb_cur=lamb_cur*2 #if chi doesn't improve, we double it
        else:
            lamb_cur = lamb_cur/(2)**0.5 #if chi improves, we divide it by sqrt(2)
            #then we update our values
            pars     = pars_new
            chi_cur  = chi_new
            resid    = resid_new
            derivs   = deriv(func,steps,pars,len(y))
        if (abs(var_chi)<1e-3) and (iter>0) : #convergence test. If chi doesn't improve much,
            # we break out of the loop
            break
    return pars, np.sqrt(np.diag(np.linalg.pinv(derivs.T@Ninv@derivs)))



plt.ion()
wmap=np.loadtxt('wmap_tt_spectrum_9yr_v5.txt')
Ninv=np.linalg.inv(np.diag(wmap[:,2])**2)

steps_derivative=np.asarray([0.108,3e-5,2.4e-4,0,1.028e-10,8.4e-4]) #les endroits où le step est nul ne e font pas deriver: ils sont fixés
pars_init=np.asarray([67.36,0.02237,0.12,0.05,2.099e-9,0.9649])
pars,pars_err=lm_camb(pars_init,steps_derivative,get_spectrum,wmap[:,1],Ninv)

steps_float_tau=np.asarray([0,0,0,1.46e-3,0,0])
pars_tau,pars_err2=lm_camb(pars,steps_float_tau,get_spectrum,wmap[:,1],Ninv) #we use the results
#from the firt lm and we only have a step on tau this time


