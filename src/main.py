import numpy                    as np
import datetime                 as dt
from pathlib import Path
from time import time
import sys
import torchode     as to
import torch
import jax.numpy as jnp

##import own scripts
import rates    as rates
import input    as input
from solve_n_save       import save, solver_scipy, solver_torchode
from ode.acodes_torch   import torchODE


print('------------------ START:', dt.datetime.now(),'---------------------')
print('')


start = time()



## ---------  read input
ρ, T, δ, Av, chemtype, rate, v, C13C12, Δt, solvertype, rate = input.setinput()
dirname, name = input.set_location()


## ---------

kB, mH, rGr, nGr, stckH, mu = input.getcst()

tic = time()

if rate == 13:
    from src.ode.dcodes     import ODE

if rate == 16:
    if solvertype == 'scipy':
        from ode.acodes     import ODE
    if solvertype == 'torch':
        from ode.acodes_torch import torchODE

timesteps = 1

## calculate H accretion on dust
Haccr = stckH *np.pi*(rGr**2.0)*ρ*nGr*(8.0*kB*T/(np.pi*mH))**0.5

## set initial conditions
n, nconsv_tot, specs, nshield_i = rates.initialise_abs(chemtype, rate)     # nconsv_tot = TOTAL in fortran code


ndot        = np.zeros(len(n))
nconsv      = np.zeros(len(nconsv_tot))


k = rates.calculate_rates(T, δ, Av, rate, nshield_i, v, C13C12)

method = 'BDF'
atol = 1.e-20
rtol = 1.e-5

## build & compile torch ODE solver
if solvertype == 'torch':
	odeterm = to.ODETerm(torchODE, with_args=True) # type: ignore
	step_method          = to.Dopri5(term=odeterm)
	step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=odeterm)
	adjoint              = to.AutoDiffAdjoint(step_method, step_size_controller) # type: ignore
	jit_solver = torch.compile(adjoint)

if solvertype == 'scipy':
    jit_solver = None




print(' >> Solving ODE for Δt =',np.round(Δt,3),'sec...')
tic = time()


## solve ODE
args = (ndot.astype(np.float64), nconsv.astype(np.float64), nconsv_tot.astype(np.float64),k.astype(np.float64), ρ, Haccr)

if solvertype == 'scipy':
    solution = solver_scipy(ODE, Δt, n, args, atol, rtol, method)
    toc = time()

    solve_time = toc-tic

    if solution['status'] != 0:
        print('Could not solve.')
        print('No solution saved, will continue with next input.')
        print(solution['message'])

        ## Save the failed model
        stop = time()
        overhead_time = (stop-start)-solve_time
        input = np.array([ρ,T,δ,Av,Δt])
        save(input, n, None, np.array([solve_time,overhead_time]), 'fail/'+str(name), k)
        print('Saved in ../out/fail/.')

        ## Restart from the previous initial abundances
        n = np.load((Path(__file__).parent / f'../out/new/{name_prev}/abundances.npy').resolve())
        
        print('------------------------------------------------------------------------------')

        

    else:
        ys = solution['y']
        ts = solution['t']

        print(solution['message'])

        print('DONE! In',np.round(solve_time,2),'seconds.')
        print('')

        stop = time()

        overhead_time = (stop-start)-solve_time

        abs = np.vstack((n,ys.T)).T
        input = np.array([ρ,T,δ,Av,Δt])

        print(' >> Saving output...')
        save(input, abs, ts, np.array([solve_time,overhead_time]), dirname+'/'+str(name), k)

        print('DONE! Output found in ../out/'+dirname+'/'+str(name)+'/')
        print('------------------------------------------------------------------------------')

        


if solvertype == 'torch':
    solution = solver_torchode(torchODE,jit_solver, Δt,n,args, atol, rtol)

    toc = time()
    solve_time = toc-tic

    print('Status solver:',solution.status)
    print('DONE! In',np.round(solve_time,2),'seconds.')
    print('')

    ys = solution.ys.data.view(-1,466).numpy()
    ts = solution.ts.data.view(-1).numpy()


    stop = time()
    overhead_time = (stop-start)-solve_time

    abs = np.vstack((n,ys)).T
    input = np.array([ρ,T,δ,Av,Δt])

    save(input, abs, ts, np.array([solve_time,overhead_time]), dirname+'/'+str(name), k)

    print('DONE! Output found in ../out/'+dirname+'/'+str(name)+'/')
    print('------------------------------------------------------------------------------')

  




print('')
print('------------------   END:', dt.datetime.now(),'---------------------')




