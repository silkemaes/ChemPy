import torch
import torchode     as to
from save import saves


def solver_torchode(ODE, jit_solver, Δt, n, args, atol, rtol):

    t_eval = np.array([0.0,Δt])

    # odeterm = to.ODETerm(ODE, with_args=True)
    # step_method          = to.Dopri5(term=odeterm)
    # step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=odeterm)
    # adjoint              = to.AutoDiffAdjoint(step_method, step_size_controller) # type: ignore

    # jit_solver = torch.compile(adjoint)

    y0     = torch.from_numpy(n     .astype(np.float64)).view((1,-1))
    t_eval = torch.from_numpy(t_eval.astype(np.float64)).view((1,-1))

    problem = to.InitialValueProblem(
        y0     = y0,        # type: ignore
        t_eval = t_eval,    # type: ignore
    )

    solution = jit_solver.solve(problem, args=args)

    return solution


def solve(ODE, Δt, n, args, atol, rtol, method, jitsolver):
    solution = solver_torchode(torchODE,jitsolver, Δt,n,args, atol, rtol)

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

    return ys[-1], name