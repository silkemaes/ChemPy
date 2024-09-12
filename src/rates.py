'''
Python code to calculate the reaction rates per reaction type:
- Two-body reaction (Arrhenius law)
- CP = direct cosmic ray ionisation
- CR = cosmic ray-induced photoreaction
- PH = Photodissociation

Written by Silke Maes, May 2023

Updated by Silke Maes to JAX, Sept 2024
'''

import numpy as np
import shielding as shield
import jax.numpy as jnp
import jax

from pathlib import Path


## Rate file handling

# reaction_type = {'AD'  : 'AD (associative detachment)',
#                  'CD'  : 'CD (collisional dissociation)',
#                  'CE'  : 'CE (charge exchange)',
#                  'CP'  : 'CP = CRP (cosmic-ray proton)',
#                  'CR'  : 'CR = CRPHOT (cosmic-ray photon)',
#                  'DR'  : 'DR (dissociative recombination)',
#                  'IN'  : 'IN (ion-neutral)',
#                  'MN'  : 'MN (mutual neutralisation)',
#                  'NN'  : 'NN (neutral-neutral)',
#                  'PH'  : 'PH (photoprocess)',
#                  'RA'  : 'RA (radiative association)',
#                  'REA' : 'REA (radiative electron attachement)',
#                  'RR'  : 'RR (radiative rec ombination)',
#                  'IP'  : 'IP (internal photon)',
#                  'AP'  : 'AP (accompaning photon)'
# }



## For numbers for faster calculation
frac = 1/300.
w = 0.5             ## grain albedo
alb = 1./(1.-w)


## Reading rate & species file


def read_rate_file():
    '''
    Read rates file (Rate12, UMIST database, including IP, AP, HNR - reactions) \n 
    (McElroy et al., 2013, M. VdS' papers)
    '''

    # print('---------------')
    # print(rate)
    # print(rate.dtype)
    # print(rate.item())
    # print('---------------')

    rate = 16
    loc = (Path(__file__).parent / f'../rates/rate{rate}.rates').resolve()
    # print(loc)

    rates = dict()
    with open(loc, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i].split(':')
            rates[int(line[0])] = line[1:]
    
    type = list()
    α = np.zeros(len(rates))
    β = np.zeros(len(rates))
    γ = np.zeros(len(rates))
    for nb in rates:
        type.append(str(rates[nb][0]))
        α[nb-1] = float(rates[nb][8])
        β[nb-1] = float(rates[nb][9])
        γ[nb-1] = float(rates[nb][10])

    print(" >> Rate file read in.")

    return jnp.array(rates), type, jnp.array(α), jnp.array(β), jnp.array(γ)


def read_specs_file(chemtype, rate):
    '''
    Read species file (Rate12, UMIST database) \n 
    (McElroy et al., 2013)\n
    '''   
    
    loc_parnt = (Path(__file__).parent / f'../rates/{chemtype}.parents').resolve()
    loc_specs = (Path(__file__).parent / f'../rates/rate{rate}.specs').resolve()
    # print(loc_parnt)

    idxs        = np.loadtxt(loc_specs, usecols=(0), dtype=int, skiprows = 1)     
    specs_all   = np.loadtxt(loc_specs, usecols=(1), dtype=str, skiprows = 1)  ## Y in fortran77 code

    specs = list()
    convs = list()
    for i in range(len(idxs)):
        idx = idxs[i]
        if idx == 0:
            convs.append(specs_all[i])
        else:
            specs.append(specs_all[i])

    parnt = np.loadtxt(loc_parnt, skiprows=0   , usecols= (0,1), dtype=str)

    return np.array(specs), parnt.T, np.array(convs)



## Setting initial abundances
def initialise_abs(chemtype, rate):
    '''
    This function sets the initial abundance of the species: 

    INPUT:
        chemtype = chemistry type: 'C' of 'O'

    RETURN:
        - abs       = abundances of non-conserved species 
        - abs_consv = abundances of conserved species 
        - specs     = array with species names 
        (The order of specs corresponds to the order of abs)
    '''
    specs, parnt, consv = read_specs_file(chemtype, rate)

    ## Initial abundances of the non-conserved species
    abs = jnp.zeros(len(specs))

    # nshield_i = dict()
    # for i in range(len(specs)):
    #     for j in range(parnt.shape[1]):
    #         if specs[i] == parnt[0][j]:
    #             abs[i] = parnt[1][j]

    #     ## store initial abundances from CO and N2, because this is needed to determine the shieldingrate
    #     if specs[i] == 'CO':
    #         nshield_i['CO'] = abs[i]
    #     elif specs[i] == 'N2':
    #         nshield_i['N2'] = abs[i]

    ## Initialise abundances of the conserved species
    abs_consv = jnp.zeros(len(consv))
    abs_consv = abs_consv.at[1].set(0.5)              ## H2
    
    abs = jnp.concatenate((abs, abs_consv))

    return abs, abs_consv, specs#, nshield_i


## Calculating the reaction rates
@jax.jit
def calculate_rates(T, δ, Av):#, rate, nshield_i, v, C13C12):
    '''
    Calculate the reaction rate for all reactions.

    First read in reaction rate file, from this, depending on the reaction type, \n 
    the correct reaction rate is calculated.
    '''
    # print(' >> Reading rate file...')
    rates, type, α, β, γ = read_rate_file()
    # print(' >> DONE!')
    # print('')

    k = np.zeros(len(type))

    # print(' >> Calculating chemical rates...')
    for i in range(len(type)):
        if type[i] == 'CP':
            k[i] = CP_rate(α[i]) 
        elif type[i] == 'CR':
            k[i] = CR_rate(α[i], β[i], γ[i], T)
        elif type[i] == 'PH':
            # if rates[i+1][1] == 'CO':
            #     print(' >> CO self-slielding...')
            #     COshieldrate = shield.retrieve_rate(nshield_i, Av, T, v, C13C12, 'CO')                
            #     k[i] = COshieldrate*photodissociation_rate(α[i], γ[i], δ, Av)
            #     print(' >> DONE!')
            #     print('')
            # elif rates[i+1][1] == 'N2':
            #     print(' >> N2 self-slielding...')
            #     N2shieldrate = shield.retrieve_rate(nshield_i, Av, T, v, None, 'N2')
            #     k[i] = N2shieldrate*photodissociation_rate(α[i], γ[i], δ, Av)
            #     print(' >> DONE!')
            #     print('')
            # else:
            k[i] = photodissociation_rate(α[i], γ[i], δ, Av)
        elif type[i] == 'IP':
            k[i] = 0
        elif type[i] == 'AP':
            k[i] = 0
        else:
            k[i] = Arrhenius_rate(α[i], β[i], γ[i], T)
    # print(' >> rates DONE!')
    # print('')

    return k



## Rate equations

@jax.jit
def Arrhenius_rate(α, β, γ, T):
    '''
    Arrhenius law for two-body reactions. \n \n
    Reaction-dependent parameters: \n
        - α = speed/probability of reaction \n
        - β = temperature dependence \n
        - γ = energy barrier \n
\n
    Physics dependent parameters:\n
        - T = temperature\n
\n
    Constants:\n
        - frac = 1/300\n
    '''
    k = α*(T*frac)**β*np.exp(-γ/T)
    return k


@jax.jit
def CP_rate(α):
    '''
    Direct cosmic ray ionisation reaction rate, give by alpha.

    For the following reaction type: CP
    '''
    k = α
    return k


@jax.jit
def CR_rate(α, β, γ, T):
    '''
    Cosmic ray-induced photoreaction rate.

    Reaction-dependent parameters:
        - α = speed/probability of reaction
        - β = temperature dependence
        - γ 
        
    Physics dependent parameters:
        - T = temperature
        - w = dust-grain albedo == 0.5

    Constants:
        - frac = 1/300
        - alb = 1./(1.-w)

    For the following reaction type: CR
    '''
    k = α * (T*frac)**β * (γ)*alb
    return k


@jax.jit
def photodissociation_rate(α, γ, δ, Av):
    '''
    For the following reaction type: PH

    Photodissociation reaction rate:
        - α = speed/probability of reaction
        - γ

    Physical parameters (input model):
        - δ = outward dilution of the radiation field
        - Av = species-specific extinction (connected to optical depth)
    '''

    AuvAv = 4.65 


    k = α * δ * np.exp(-γ * Av/AuvAv)
    return k







        
