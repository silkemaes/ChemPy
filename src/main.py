import numpy                    as np
import datetime                 as dt
from pathlib import Path
from time import time

##import own scripts
import rates    as rates
import input    as input


print('------------------ START:', dt.datetime.now(),'---------------------')
print('')


kB, mH, rGr, nGr, stckH = input.getcst()

tic = time()
## input
ρ, T, δ, Av, chemtype, rate, v, C13C12 = input.setinput()

if rate == '13':
    import ode.dcodes   as odes
if rate == '16':
    import ode.acodes   as odes
import sys
timesteps = 1

## calculate H accretion on dust
Haccr = stckH *np.pi*(rGr**2.0)*ρ*nGr*(8.0*kB*T/(np.pi*mH))**0.5

## set initial conditions
n, nconsv_tot, specs, nshield_i = rates.initialise_abs(chemtype, rate)     # nconsv_tot = TOTAL in fortran code
timesteps = 1


ndot        = np.zeros(len(n))
nconsv      = np.zeros(len(nconsv_tot))
t           = np.zeros(timesteps)


k = rates.calculate_rates(T, δ, Av, rate, nshield_i, v, C13C12)

ndot,X = odes.ODE(t, n, ndot, nconsv, nconsv_tot,k, ρ, Haccr)
toc = time()

print('Time elapsed: ',toc-tic)

print(ndot.shape, X.shape)

## Logging the runs of main.py
# filename = 'log'
# loc = (Path(__file__).parent / f'../{filename}.txt').resolve()
# extra_message = ''

# with open(loc, 'a') as f:
#     f.write('\nDate: '+str(dt.datetime.now())+'\n\n')
#     f.write('Input:\n\n')
#     f.write('   ρ  = '+'{:.2E}'.format(ρ)+'\n')
#     f.write('   v  = '+str(v)+'\n')
#     f.write('   T  = '+str(T)+'\n')
#     f.write('   δ  = '+str(δ)+'\n')
#     f.write('   Av = '+str(Av)+'\n')
#     f.write('   Chem type = '+chemtype+'\n')
#     f.write('   Rate      = '+rate+'\n\n')
#     f.write('Info:\n')
#     f.write('   '+extra_message+'\n')
#     f.write('\n--------------------------------\n')





print('')
print('------------------   END:', dt.datetime.now(),'---------------------')




