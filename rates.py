'''
Python code to calculate the reaction rates per reaction type:
- Two-body reaction (Arrhenius law)
- CP = direct cosmic ray ionisation
- CR = cosmic ray-induced photoreaction
- photodissociation

Written by Silke Maes, May 2023
'''

import numpy as np
from numba import njit


## Rate file handling

reaction_type = {'AD'  : 'AD (associative detachment)',
                 'CD'  : 'CD (collisional dissociation)',
                 'CE'  : 'CE (charge exchange)',
                 'CP'  : 'CP = CRP(cosmic-ray proton)',
                 'CR'  : 'CR = CRPHOT(cosmic-ray photon)',
                 'DR'  : 'DR (dissociative recombination)',
                 'IN'  : 'IN (ion-neutral)',
                 'MN'  : 'MN (mutual neutralisation)',
                 'NN'  : 'NN (neutral-neutral)',
                 'PH'  : 'PH (photoprocess)',
                 'RA'  : 'RA (radiative association)',
                 'REA' : 'REA (radiative electron attachement)',
                 'RR'  : 'RR (radiative rec ombination)',
                 'IP'  : 'IP (internal photon)',
                 'AP'  : 'AP (accompaning photon)'
}

## dit klopt niet helemaal
ranges_type = {'AD'  : [0,131]    ,
               'CD'  : [132,145]  ,
               'CE'  : [146,724]  ,
               'CP'  : [725,735]  ,
               'CR'  : [736,984]  ,
               'DR'  : [985,1515] ,
               'IN'  : [1516,4104],
               'MN'  : [4105,5085],
               'NN'  : [5086,5704],
               'PH'  : [5705,6040],
               'RA'  : [6041,6132],
               'REA' : [6133,6156],
               'RR'  : [6157,6172],
}

source_type = { 'E' : 'estimated',
                'M' : 'measured',
                'C' : 'calculated',
                'L' : 'literature (experimental)'
              }

accuracy    = { 'A' : '< 0.25',
                'B' : '< 0.50',
                'C' : 'within factor of 2',
                'D' : 'within order of magn',
                'E' : 'highly uncertain'
              }


ding = 1/300.


## Rate equations

'''
Arrhenius law for two-body reactions.
Reaction-dependent parameters:
    - α = speed/probability of reaction
    - β = temperature dependence
    - γ = energy barrier
Physics dependent parameters:
    - T = temperature

For the following reaction types:
    - 

'''
@njit
def Arrhenius_rate(α, β, γ, T):
    k = α*(T*ding)**β*np.exp(-γ/T)
    return k

'''
Direct cosmic ray ionisation reaction rate, give by alpha.

For the following reaction type: CP
'''
@njit
def CP_rate(α):
    k = α
    return k

'''
Cosmic ray-induced photoreaction rate.
Reaction-dependent parameters:
    - α = speed/probability of reaction
    - β = temperature dependence
    - γ 
Physics dependent parameters:
    - T = temperature
    - w = dust-grain albedo

For the following reaction type: CR
'''
@njit
def CR_rate(α, β, γ, T, w):
    k = α * (T*ding)**β * (γ)/(1.-w)
    return k

'''
Photodissociation reaction rate.
    - α = speed/probability of reaction
    - γ
Physical parameters:
    - δ = overall dilution of the radiation field
    - Av = species-specific extinction (connected to optical depth)

For the following reaction type: PH
'''
@njit
def photodissociation_rate(α, γ, δ, Av):
    k = α * δ * np.exp(-γ * Av)
    return k


    