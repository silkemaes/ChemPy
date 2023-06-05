import numpy                    as np
import datetime                 as dt
import sys

##import own scripts
import rates        as rates
import input    as input


print('------------------ START:', dt.datetime.now(),'---------------------')
print('')


kB, mH, rGr, nGr, stckH, AUV_AV = input.getcst()


## input
ρ, T, δ, Av, chemtype, rate = input.setinput()

if rate == '13':
    import ode.dcodes   as odes
if rate == '16':
    import ode.acodes   as odes

timesteps = 1

## calculate H accretion on dust
Haccr = stckH *np.pi*(rGr**2.0)*ρ*nGr*(8.0*kB*T/(np.pi*mH))**0.5

## set initial conditions
n, nconsv_tot, specs = rates.initialise_abs(chemtype, rate)     # nconsv_tot = TOTAL in fortran code
timesteps = 1


ndot        = np.zeros(len(n))
nconsv      = np.zeros(len(nconsv_tot))
t           = np.zeros(timesteps)

k = rates.calculate_rates(T, δ, Av, rate)

ndot = odes.ODE(t, n, ndot, nconsv, nconsv_tot,k, ρ, Haccr)











print('')
print('------------------   END:', dt.datetime.now(),'---------------------')




