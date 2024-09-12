def save(input, abs, ts, time, name, k):
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
    np.save((Path(__file__).parent / f'../{loc}rates').resolve(), k)
    
    return