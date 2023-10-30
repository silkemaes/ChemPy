import numpy as np
import os

class ChemTorchMod():
    '''
    Object representing 1 ChemTorch model.
    Contains:
        - n     [2d np.array]: Abundances at different timesteps
        - tstep [1d np.array]: Timesteps that the classical ODE solver is evaluated
        - p     [1d np.array]: input of the model -> [rho, T, delta, Av]
    '''
    def __init__(self, dirname, dir=None):
        outpath = '/STER/silkem/ChemTorch/out/'
        
        self.n      = np.load(outpath+dirname+'/'+dir+'/abundances.npy')[:,1:] # type: ignore
        self.tstep  = np.load(outpath+dirname+'/'+dir+'/tstep.npy') # type: ignore
        input       = np.load(outpath+dirname+'/'+dir+'/input.npy') # type: ignore
        self.p      = input[0:-1]
        self.tictoc = np.load(outpath+dirname+'/'+dir+'/tictoc.npy') # type: ignore

    def __len__(self):
        return len(self.tstep)


class CSEmod():
    '''
    Class to initialise the dataset to train & test emulator

    Get data from textfiles (output CSE model)
    
    Preprocess:
        - set all abundances < cutoff to cutoff
        - take np.log10 of abudances

    '''
    def __init__(self, loc, dir=None, file=None):
        data = []

        if dir != None:
            locs = os.listdir(dir) 

            for i in range(1,len(locs)+1):
                name = dir+'csfrac_smooth_'+str(i)+'.out'
                proper = read_data_1Dmodel(name)
                data.append(proper)
        
        if file != None:
            proper = read_data_1Dmodel(file)
            data.append(proper)

        df = np.concatenate(data)

        self.n = df

def read_data_1Dmodel(file_name):
    '''
    Read data text file of output abundances of 1D CSE models
    '''
    with open(file_name, 'r') as file:
        dirty = []
        proper = None
        for line in file:
            try:  
                if len(line) > 1: 
                    dirty.append([float(el) for el in line.split()])
            except:
                if len(dirty) != 0:
                    dirty = np.array(dirty)[:,1:]
                    if proper is None:
                        proper = dirty
                    else:
                        proper = np.concatenate((proper, dirty), axis = 1)
                dirty = []
    return proper