import numpy            as np

from time               import time
from pathlib            import Path
import datetime         as dt
import os

import rates        as rates
from input          import getcst

import jax.numpy as jnp




def solve(input, Δt, rate, n,  nconsv_tot, name_prev ,dirname, solvertype,jitsolver, method = 'BDF',atol = 1.e-20, rtol = 1.e-5):
    '''
    Solve the chemical ODE, given by the ODE function. \n
    Adjusted for data generation process \n
    \n
    INPUT: \n
        - input = np.array(ro,T,delta,Av) \n
        - Δt = solve over this amount of seconds \n
        - rate = integer, gives version of the rate equations and ODE, either 13 or 16. 16 is prefered. \n

    '''
    name = dt.datetime.now()

    start = time()

    ρ  = input[0]
    T  = input[1]
    δ  = input[2]
    Av = input[3]

    print('------------------------------------------------------------------------------')
    print('Directory & Name:')
    print(dirname,' / ' ,name)
    print('Start with abundances from',name_prev)
    print('')
    print('Input:')
    print('[density, temperature, delta, Av] dt:')
    print(input,np.round(Δt,2))
    print('Solver type:', solvertype)
    print('')

    if rate == 13:
        from ode.dcodes     import ODE

    if rate == 16:
        if solvertype == 'scipy':
            from ode.acodes     import ODE
            from ode.acodes     import calc_conserved
        if solvertype == 'torch':
            from ode.acodes_torch import torchODE

    kB, mH, rGr, nGr, stckH, mu = getcst()    

    ## calculate H accretion on dust
    Haccr = stckH *np.pi*(rGr**2.0)*ρ*nGr*(8.0*kB*T/(np.pi*mH))**0.5

    ndot        = jnp.zeros(len(n))
    nconsv      = jnp.zeros(len(nconsv_tot))

    v = 11
    C13C12 = 69  ## Ramstedt & Olofsson (2014)

    k = rates.calculate_rates(T, δ, Av)#, rate, nshield_i, v, C13C12)
    

    print(' >> Solving ODE for Δt =',np.round(Δt,3),'sec...')
    tic = time()
    ## solve ODE
    args = (ndot, nconsv, nconsv_tot,k, ρ, Haccr)

    if solvertype == 'scipy':
        import solver_scipy as sol
    
    if solvertype == 'torch':
        import solver_torch as sol

    if solvertype == 'jax':
        import solver_jax as sol

    y,name = sol.solve(ODE, Δt, n, args, atol, rtol, method, jitsolver)        

    return y, name



