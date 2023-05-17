import numpy as np
from astropy import constants as cst


## Physical constants
kB = cst.k_B.cgs.value          ## Boltzmann constant [erg/K]
mH = cst.u.to('g').value        ## mass H atom = atomic mass unit [g]

## Other constants
pi = np.pi


#-- GRAIN PARAMETERS FOR H2 FORMATION AND CR IONISATION
rGr = 1.0E-5        ## grain radius [cm] (A_G in fortran77 code)
nGr = 1.5e-12       ## grain number density/H2 (assuming gas/dust = 200, rho = 3.5 g/cm^3) (X_G in fortran77 code)
w = 0.5             ## grain albedo
ding2 = 1.-w
γ_CO = 3.           ## (GAMCO in fortran77)
AUV_AV = 4.65
stckH = 0.3         ## sticking coefficient for H atoms



def read_rate_file():

    loc = 'rates/rate16_IP_2330K_AP_6000K.rates'

    rates = dict()
    with open(loc, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i].split(':')
            rates[int(line[0])] = line[1:]
    
    type = list()
    α = list()
    β = list()
    γ = list()
    for rate in rates:
        type.append((rates[rate][0]))
        α.append(float(rates[rate][8]))
        β.append(float(rates[rate][9]))
        γ.append(float(rates[rate][10]))

    return rates, type, np.array(α), np.array(β), np.array(γ)


## input
ρ = 1e-6
T = 2500.
δ = 1.
Av = 1.
    

Haccr = stckH *pi*(rGr**2.0)*ρ*nGr*(8.0*kB*T/(pi*mH))**0.5



