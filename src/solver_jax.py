
import diffrax
import jax

jax.config.update("jax_enable_x64", True)

@jax.jit
def solver_jax(ODE, Δt, n, args, atol, rtol, method):

    print('... in solver loop ...')

    terms = diffrax.ODETerm(ODE, args=args)
    t0 = 0.0
    t1 = Δt
    y0 = n

    solver = diffrax.Kvaerno5()   
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        y0,
        stepsize_controller=stepsize_controller,
    )

    return sol

@jax.jit
def solve(ODE, Δt, n, args, atol, rtol, method, jitsolver):
    
        solution = solver_jax(ODE, Δt, n, args, atol, rtol, method)
        toc = time()
    
        solve_time = toc-tic
    
        print('DONE! In',np.round(solve_time,2),'seconds.')
        print('')
    
        ys = solution.ys
        ts = solution.ts

        print(ys.shape, ys)
    
        stop = time()
    
        overhead_time = (stop-start)-solve_time
    
        abs = np.vstack((n,ys)).T
        input = np.array([ρ,T,δ,Av,Δt])
    
        save(input, abs, ts, np.array([solve_time,overhead_time]), dirname+'/'+str(name), k)
    
        print('DONE! Output found in ../out/'+dirname+'/'+str(name)+'/')
        print('------------------------------------------------------------------------------')
    
        return ys[-1], name