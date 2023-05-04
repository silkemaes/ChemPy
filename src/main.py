import numpy as np
from astropy import constants as cst


## Physical constants
kB = cst.k_B.cgs.value          ## Boltzmann constant [erg/K]
mH = cst.u.to('g').value        ## mass H atom = atomic mass unit [g]

## Other constants
pi = np.pi


#-- GRAIN PARAMETERS FOR H2 FORMATION AND CR IONISATION
rGr = 1.0E-5        ## grain radius [cm]
nGr = 1.5e-12       ## grain number density/H2 (assuming gas/dust = 200, rho = 3.5 g/cm^3)
w = 0.5             ## grain albedo
AUV_AV = 4.65
stckH = 0.3         ## sticking coefficient for H atoms



def readrates(chemtype):
    if chemtype == 'C':
        ratefile = '/lhome/silkem/ChemTorch/ChemTorch/rates/rate16_IP_6000K_Crich_mean_Htot.specs'

    return

def readinput(file):
    

    Haccr = stckH *pi*(rGr**2.0)*ρ*nGr*(8.0*kB*T/(pi*mH))**0.5
    return ρ, T, δ, Av, Haccr


