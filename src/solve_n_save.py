import numpy            as np
from time               import time
from pathlib            import Path
import datetime         as dt

import src.rates        as rates
from src.input          import getcst

from scipy.integrate    import solve_ivp
from astropy            import units

def solve(ρ, T, δ, Av, v, C13C12, chemtype, Δt, rate, filename, logmessage = '', method = 'BDF',atol = 1.e-30, rtol = 1.e-7, getydot = False):
    '''
    Solve the chemical ODE, given by the ODE function. \n
    Returns t, y, ydot.
    '''

    if rate == 13:
        from src.ode.dcodes     import ODE
    if rate == 16:
        from src.ode.acodes     import ODE

    kB, mH, rGr, nGr, stckH = getcst()
    yr_to_sec = units.year.to('s')    

    ## calculate H accretion on dust
    Haccr = stckH *np.pi*(rGr**2.0)*ρ*nGr*(8.0*kB*T/(np.pi*mH))**0.5

    ## set initial conditions
    n, nconsv_tot, specs, nshield_i = rates.initialise_abs(chemtype, rate)     # nconsv_tot = TOTAL in fortran code
    timesteps = 1

    ndot        = np.zeros(len(n))
    nconsv      = np.zeros(len(nconsv_tot))
    t           = np.zeros(timesteps)

    k = rates.calculate_rates(T, δ, Av, rate, nshield_i, v, C13C12)

    Δt = Δt*yr_to_sec  ## in sec


    ## Logging the runs of main.py
    loc = 'log_tests.txt'
    with open(loc, 'a') as f:
        f.write('\nDate: '+str(dt.datetime.now())+'\n\n')
        f.write('Input:\n\n')
        f.write('   ρ  = '+'{:.2E}'.format(ρ)+'\n')
        f.write('   v  = '+str(v)+'\n')
        f.write('   T  = '+str(T)+'\n')
        f.write('   δ  = '+str(δ)+'\n')
        f.write('   Av = '+str(Av)+'\n')
        f.write('   13C/12C   = '+str(C13C12)+'\n')
        f.write('   Chem type = '+chemtype+'\n')
        f.write('   Rate      = '+str(rate)+'\n\n')
        f.write('Output file: '+filename+'\n\n')
        f.write('Info:\n')
        f.write('   '+logmessage+'\n')
        f.write('\n--------------------------------\n')

    # solvers = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']

    print(' >> Solving ODE with Δt',Δt,'...')
    tic = time()
    ## solve ODE
    solution = solve_ivp(
        fun          = ODE,
        t_span       = [0.0, Δt],
        y0           = n.astype(np.float64),
        method       = method,   ## zoals DVODE
        args         = (ndot, nconsv, nconsv_tot,k, ρ, Haccr),
        dense_output = False,    ## if True: returns interpolation function
        atol         = atol,
        rtol         = rtol
        )
    toc = time()

    assert solution['status'] == 0

    ys = solution['y']
    ts = solution['t']

    print(solution['message'])

    if getydot == 'True':
        print(' >> Calculating ydot for solution...')
        ydot = [ODE(t, ys[:,i], ndot, nconsv, nconsv_tot,k, ρ, Haccr) for i in range(ys.shape[1])]

        print('DONE!')
        print('')

        print('>> Saving output...')

        save(ts, ys, specs, filename)

        return ts, ys, ydot, toc-tic, specs

    else:
        print('DONE!')
        print('')

        print('>> Saving output...')

        save(ts, ys, specs, filename)

        print('DONE!')

        return ts, ys,  toc-tic, specs


def save(ts, ys, specs, filename):

    out = np.array(ys)

    yr_to_sec = units.year.to('s') 

    loc = (Path(__file__).parent / f'../out/{filename}.out').resolve() 

    with open(loc, 'w') as f:
        f.write('{0:11}'.format('TIME [yr]'))
        for spec in specs:
            f.write('{0:11}'.format(spec))
        f.write('\n')
        for i in range(out.shape[1]):
            f.write('{:.12e}'.format(ts[i]/yr_to_sec)+'  ')
            for j in range(out.shape[0]):
                f.write('{:.12e}'.format(out[j][i])+'  ')
            f.write('\n')

    return    