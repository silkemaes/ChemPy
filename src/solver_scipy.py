from scipy.integrate    import solve_ivp
from save import save



def solver_scipy(ODE, Δt, n, args, atol, rtol, method):

    # print(Δt, Δt.dtype)

    # solver_methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']

    print('... in solver loop ...')

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


def solve(ODE, Δt, n, args, atol, rtol, method, jitsolver):

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

        return n.T[0], name_prev

    else:
        ys = solution['y']
        ts = solution['t']

        print(solution['message'])
        # print(nconsv_tot)

        print('DONE! In',np.round(solve_time,2),'seconds.')
        print('')

        stop = time()

        overhead_time = (stop-start)-solve_time

        abs = np.concatenate((n,ys.T[-1]))
        input = np.array([ρ,T,δ,Av,Δt])

        print(' >> Saving output...')
        save(input, abs, ts, np.array([solve_time,overhead_time]), dirname+'/'+str(name), k)

        print('DONE! Output found in ../out/'+dirname+'/'+str(name)+'/')
        print('------------------------------------------------------------------------------')

        return ys.T[-1], name