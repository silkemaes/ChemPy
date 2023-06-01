import numpy                    as np
import datetime                 as dt
import sys

##import own scripts
import rates    as rates
import odes     as odes
import input    as input


print('------------------ START:', dt.datetime.now(),'---------------------')
print('')


kB, mH, rGr, nGr, stckH, AUV_AV = input.getcst()


## input
ρ, T, δ, Av, chemtype = input.setinput()

timesteps = 1

## calculate H accretion on dust
Haccr = stckH *np.pi*(rGr**2.0)*ρ*nGr*(8.0*kB*T/(np.pi*mH))**0.5

## set initial conditions
n, nconsv_tot, specs = rates.initialise_abs(chemtype)     # nconsv_tot = TOTAL in fortran code
timesteps = 1

ndot        = np.zeros(len(n))
nconsv      = np.zeros(len(nconsv_tot))
t           = np.zeros(timesteps)

k = rates.calculate_rates(T, δ, Av)

ndot = odes.ODE(t, n, ndot, nconsv, nconsv_tot,k, ρ, Haccr)











print('')
print('------------------   END:', dt.datetime.now(),'---------------------')




