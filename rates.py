'''
Python code to calculate the reaction rates per reaction type:
- Two-body reaction (Arrhenius law)
- CP = direct cosmic ray ionisation
- CR = cosmic ray-induced photoreaction
- photodissociation

Written by Silke Maes, May 2023
'''




import numpy as np


'''
Arrhenius law for two-body reactions.
Reaction-dependent parameters:
    - alpha = speed/probability of reaction
    - beta = temperature dependence
    - gamma = energy barrier
Physics dependent parameters:
    - T = temperature
'''
def Arrhenius_rate(α, β, γ, T):
    k = α*(T/300)**β*np.exp(-γ/T)
    return k

'''
Direct cosmic ray ionisation reaction rate, give by alpha.
'''
def CP_rate(α):
    k = α
    return k

'''
Cosmic ray-induced photoreaction rate.
Reaction-dependent parameters:
    - alpha = speed/probability of reaction
    - beta = temperature dependence
    - gamma 
Physics dependent parameters:
    - T = temperature
    - w = dust-grain albedo
'''
def CR_rate(α, β, γ, T, w):
    k = α * (T/300)**β * (γ)/(1-w)
    return k

'''
Photodissociation reaction rate.
    - alpha = speed/probability of reaction
    - gamma 
Physical parameters:
    - delta = overall dilution of the radiation field
    - Av = species-specific extinction (connected to optical depth)
'''
def photodissociation_rate(α, γ, δ, Av):
    k = α * δ * np.exp(-γ * Av)
    return k


    