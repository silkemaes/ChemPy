import numpy                    as np
from astropy import constants   as cst
import datetime                 as dt
import sys

##import own scripts
sys.path.insert(1, '/lhome/silkem/ChemTorch/ChemTorch/src/ode')
import rates    as rates
import odes     as odes


print('------------------ START:', dt.datetime.now(),'---------------------')
print('')


## Physical constants
kB = cst.k_B.cgs.value          ## Boltzmann constant [erg/K]
mH = cst.u.to('g').value        ## mass H atom = atomic mass unit [g]

## Other constants
pi = np.pi


## Grain parameters for H2 formation and cosmic ray ionisation
rGr = 1.0E-5        ## grain radius [cm] (A_G in fortran77 code)
nGr = 1.5e-12       ## grain number density/H2 (assuming gas/dust = 200, rho = 3.5 g/cm^3) (X_G in fortran77 code)
w = 0.5             ## grain albedo
γ_CO = 3.           ## (GAMCO in fortran77)
AUV_AV = 4.65
stckH = 0.3         ## sticking coefficient for H atoms


## input values physics
ρ = 1.e+6
T = 2500.

δ = 1.      ## = RAD in fortran; outwards dilution
Av = 1.     ## outwards dust extinction 

r = 1.        ## setting inwards geometrical dilution
ΔAv = 1.      ## inwards dust extinction 


timesteps = 1

## input chemistry
chemtype = 'C'

## calculate H accretion on dust
Haccr = stckH *pi*(rGr**2.0)*ρ*nGr*(8.0*kB*T/(pi*mH))**0.5


n, n_consv = rates.initialise_abs(chemtype)     # n_consv = TOTAL in fortran code

ndot = np.zeros(len(n))
X    = np.zeros(len(n_consv))
t    = np.zeros(timesteps)

k = rates.calculate_rates(T, δ, Av)


X, ndot, n = odes.ODE(t, n, ndot, X, n_consv,k, ρ, Haccr)











print('')
print('------------------   END:', dt.datetime.now(),'---------------------')




