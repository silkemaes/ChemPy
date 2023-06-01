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
        - density in units of g/cm^3
    '''
    # r    = 1e18 #* unt.cgs.cm                       # cm
    Mdot = Mdot * Msunyr                            # gram/s
    vexp = v    * cms                               # cm/s


    dens = Mdot / (4*np.pi * vexp * r**2 * mu * mH)       # g/cm^3

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
    ρ = density(1.e-7, 10., 1.e15)
    # ρ = 1.e+6
    T = 2500.
    δ = 1.      
    Av = 1.   

    ## input chemistry
    chemtype = 'O'

    r = 1.        ## setting inwards geometrical dilution
    ΔAv = 1.      ## inwards dust extinction 

    print('Input:')
    print('------')
    print('ρ  =','{:.2E}'.format(ρ))
    print('T  =',T)
    print('δ  =',δ)
    print('Av =',Av)
    print('')
    print('Chem type =', chemtype)

    return ρ, T, δ, Av, chemtype

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
    γ_CO = 3.           ## (GAMCO in fortran77)
    AUV_AV = 4.65
    stckH = 0.3         ## sticking coefficient for H atoms

    return kB, mH, rGr, nGr, stckH, AUV_AV