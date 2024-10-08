from astropy import constants   as cst
from astropy import units       as units
import numpy as np


## Physical constants
kB = cst.k_B.cgs.value          ## Boltzmann constant [erg/K]
mH = cst.u.to('g').value        ## mass H atom = atomic mass unit [g]
Msun = cst.M_sun.cgs.value      ## gram
yr = units.year.to('s')         ## year in seconds
Msunyr = Msun/yr                ## units of mass-loss rate in gram/s
cms  = 1e5                      ## units of velocity in cm/s
g_to_kg = units.gram.to('kg')
cm_to_m = units.cm.to('m')

mu = 2.0 + 4.0*0.17             ## mu (average mass per H2 molecule), taking into account the abundance of He


def density(Mdot,v, r):
    '''
    Input 
        - mass-loss rate (Mdot) in units of Msol/yr
        - outflow velocity (v) in units of km/s
        - radius (r): location of the outflow, inputs of cm
    Output
        - number density in units of cm^-3
    '''
    # r    = 1e18 #* unt.cgs.cm                       # cm
    Mdot = Mdot * Msunyr                            # gram/s
    vexp = v    * cms                               # cm/s

    dens = Mdot / (4*np.pi * vexp * r**2 * mu * mH)       # cm^-3

    # dens = dens * g_to_kg * cm_to_m**(-3)           # kg/m^3

    return dens

## input values physics
def setinput():
    '''
    Set input values of the model.

    INPUT:
        - ρ  = density       [g/cm^3]
        - T  = temperature   [K]
        - δ  = outwards dilution of radiation field     == RAD
        - Av = outward dus extinction
        - chemtype = type of chemistry: 'C' or 'O'
    '''

    ## input physics
    Mdot = 1.e-7    ## mass-loss rate in Msol/yr
    vexp = 10.      ## outflow velocity in km/s
    r    = 1.e14    ## location of the outflow in cm
    ρ = density(Mdot, vexp, r)
    T = 2500.       ## temperature in K
    ##- Radiation parameters, see https://ui.adsabs.harvard.edu/abs/2024ApJ...969...79M/abstract
    δ = 1.e-5      
    Av = 0.05  
    ##-
    Δt = 50000.     ## time step in seconds
    solvertype = 'torch'
    rate = 16

    r = 1.        ## setting inwards geometrical dilution
    ΔAv = 1.      ## inwards dust extinction  

    ## Temporary for testing the shielding
    C13C12 = 35


    ## input chemistry
    chemtype = 'O'

    ## choose rate-file 13, 16
    rate = 16

    print('-----------------------')
    print('| Input:')
    print('|    ')
    print('|    ρ  =','{:.2E}'.format(ρ))
    print('|    T  =',T)
    print('|    δ  =',δ)
    print('|    Av =',Av)
    print('|    Chem type =', chemtype)
    print('|    Rate      =', rate)
    print('-----------------------')
    print('')

    return ρ, T, δ, Av, chemtype, str(rate), vexp, C13C12, Δt, solvertype, rate

def set_location():
    dirname = 'test/'
    name = 'test_torch'
    return dirname, name

def getcst():
    '''
    Function to get needed physical constants & numbers.
    '''

    ## Physical constants
    kB = cst.k_B.cgs.value          ## Boltzmann constant [erg/K]
    mH = cst.u.to('g').value        ## mass H atom = atomic mass unit [g]


    ## Grain parameters for H2 formation and cosmic ray ionisation
    rGr = 1.0E-5        ## grain radius [cm] (A_G in fortran77 code)
    nGr = 1.5e-12       ## grain number density/H2 (assuming gas/dust = 200, rho = 3.5 g/cm^3) (X_G in fortran77 code)
    # γ_CO = 3.           ## (GAMCO in fortran77)
    # AUV_AV = 4.65
    stckH = 0.3         ## sticking coefficient for H atoms

    mu = 2.0 + 4.0*0.17             ## mu (average mass per H2 molecule), taking into account the abundance of He


    return kB, mH, rGr, nGr, stckH, mu