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

n, nconsv, specs = rates.initialise_abs(chemtype)     # n_consv = TOTAL in fortran code

ndot        = np.zeros(len(n))
nconsvdot   = np.zeros(len(nconsv))
t           = np.zeros(timesteps)

k = rates.calculate_rates(T, δ, Av)

ndot = odes.ODE(t, n, ndot, nconsvdot, nconsv,k, ρ, Haccr)











print('')
print('------------------   END:', dt.datetime.now(),'---------------------')




