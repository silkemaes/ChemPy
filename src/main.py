import numpy as np
from astropy import constants as cst


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
ρ = 1e-6
T = 2500.
δ = 1.
Av = 1.

## input chemistry
chemtype = 'C'


## read rates-file
rates, reaction_type, α, β, γ = read_rate_file()
## read species_file
specs, parnt, consv = read_specs_file(chemtype)


## calculate H accretion on dust
Haccr = stckH *pi*(rGr**2.0)*ρ*nGr*(8.0*kB*T/(pi*mH))**0.5



