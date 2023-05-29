from astropy import constants   as cst
import numpy as np



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
    ρ = 1.e+6
    T = 2500.
    δ = 1.      
    Av = 0.5   

    ## input chemistry
    chemtype = 'C'

    r = 1.        ## setting inwards geometrical dilution
    ΔAv = 1.      ## inwards dust extinction 

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