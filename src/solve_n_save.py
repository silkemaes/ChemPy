import numpy            as np
import torch
import torchode     as to
from time               import time
from pathlib            import Path
import datetime         as dt
import os

import src.rates        as rates
from src.input          import getcst

from scipy.integrate    import solve_ivp
from astropy            import units



def solver_scipy(ODE, Δt,n,args, atol, rtol, method):

    # solver_methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']

    solution = solve_ivp(
        fun          = ODE,
        t_span       = [0.0, Δt],
        y0           = n.astype(np.float64),    ## hier terug abundanties meegeven
        method       = method,                  ## zoals DVODE
        args         = args,
        dense_output = False,                   ## if True: returns interpolation function
        atol         = atol,
        rtol         = rtol
        )
   
    return solution

def solver_torchode(ODE, jit_solver, Δt, n, args, atol, rtol):

    t_eval = torch.from_numpy(np.array([0.0,Δt]))

    # odeterm = to.ODETerm(ODE, with_args=True)
    # step_method          = to.Dopri5(term=odeterm)
    # step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=odeterm)
    # adjoint              = to.AutoDiffAdjoint(step_method, step_size_controller) # type: ignore

    # jit_solver = torch.compile(adjoint)

    problem = to.InitialValueProblem(
        y0     = torch.from_numpy(n).view((1,-1)),   # type: ignore
        t_eval = t_eval.view((1,-1)), # type: ignore
    )

    solution = jit_solver.solve(problem, args=args)

    return solution


def solve(input, Δt, rate, n, nshield_i, nconsv_tot, name_prev ,dirname, solvertype,jitsolver, method = 'BDF',atol = 1.e-20, rtol = 1.e-5):
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
        from src.ode.dcodes     import ODE

    if rate == 16:
        if solvertype == 'scipy':
            from src.ode.acodes     import ODE
        if solvertype == 'torch':
            from src.ode.acodes_torch import torchODE

    kB, mH, rGr, nGr, stckH = getcst()    

    ## calculate H accretion on dust
    Haccr = stckH *np.pi*(rGr**2.0)*ρ*nGr*(8.0*kB*T/(np.pi*mH))**0.5

    ndot        = np.zeros(len(n))
    nconsv      = np.zeros(len(nconsv_tot))

    v = 11
    C13C12 = 69  ## Ramstedt & Olofsson (2014)

    k = rates.calculate_rates(T, δ, Av, rate, nshield_i, v, C13C12)
    

    print(' >> Solving ODE for Δt =',np.round(Δt,3),'sec...')
    tic = time()
    ## solve ODE
    args = (ndot, nconsv, nconsv_tot,k, ρ, Haccr)

    if solvertype == 'scipy':
        solution = solver_scipy(ODE, Δt,n,args, atol, rtol, method)
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
            save(input, n, None, np.array([solve_time,overhead_time]), 'fail/'+str(name))
            print('Saved in ../out/fail/.')

            ## Restart from the previous initial abundances
            n = np.load((Path(__file__).parent / f'../out/new/{name_prev}/abundances.npy').resolve())
            
            print('------------------------------------------------------------------------------')

            return n.T[0], name_prev

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
            save(input, abs, ts, np.array([solve_time,overhead_time]), dirname+'/'+str(name))

            print('DONE! Output found in ../out/'+dirname+'/'+str(name)+'/')
            print('------------------------------------------------------------------------------')

            return ys.T[-1], name 


    if solvertype == 'torch':
        solution = solver_torchode(torchODE,jitsolver, Δt,n,args, atol, rtol)

        toc = time()
        solve_time = toc-tic

        print('DONE! In',np.round(solve_time,2),'seconds.')
        print('')

        ys = solution.ys.data.view(-1,466).numpy()
        ts = solution.ts.data.view(-1).numpy()


        stop = time()
        overhead_time = (stop-start)-solve_time

        abs = np.vstack((n,ys)).T
        input = np.array([ρ,T,δ,Av,Δt])

        save(input, abs, ts, np.array([solve_time,overhead_time]), dirname+'/'+str(name))

        print('DONE! Output found in ../out/'+dirname+'/'+str(name)+'/')
        print('------------------------------------------------------------------------------')

        return ys[-1], name


def save(input, abs, ts, time, name):
    '''
    Save model input & output as '.npy' object. \n
    - input = 1D np.array(rho, T, delta, Av, dt) \n
    - abs   = 2D np.array([initial abundances],[final abunances]) \n
    - time  = 1D np.array(time needed to solve the ODE system, overhead time) \n
    - name  = name of the model = datetime.now()
    '''

    loc = 'out/'+str(name)+'/' 
    newpath = (Path(__file__).parent / f'../{loc}').resolve()
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    np.save((Path(__file__).parent / f'../{loc}input').resolve(), input) 
    np.save((Path(__file__).parent / f'../{loc}abundances').resolve(), abs)
    np.save((Path(__file__).parent / f'../{loc}tstep').resolve(), ts)
    np.save((Path(__file__).parent / f'../{loc}tictoc').resolve(), time)
    
    return

