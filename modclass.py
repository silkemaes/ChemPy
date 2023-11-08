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
    def __init__(self, loc, dir=None, modelname=None):

        if loc == 'STER':
            outloc = '/STER/silkem/CSEchem/'
            outputdir = dir+'-'+modelname
            inputfile = outputdir+'/inputChemistry_'+modelname+'.txt'
        if loc == 'home':
            outloc = '/lhome/silkem/CHEM/'
            outputdir = dir+'/'+modelname
            inputfile = dir+'/inputChemistry_'+modelname+'.txt'

        ## retrieve input
        self.Rstar, self.Tstar, self.Mdot, self.v, self.eps, self.rtol, self.atol = read_input_1Dmodel(inputfile)



        ## retrieve abundances
        abs = read_data_1Dmodel(outputdir+'/csfrac_smooth.out')

        self.n = abs

        ## retrieve physical parameters
        arr = np.loadtxt(outputdir+'/csphyspar_smooth.out', skiprows=4, usecols=(0,1,2,3,4))
        self.radius, self.dens, self.temp, self.Av, self.delta = arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4]
        self.time = self.radius/(self.v) 



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


def read_input_1Dmodel(file_name):
    with open(file_name) as file:
        lines = file.readlines()
        lines = [item.rstrip() for item in lines]

    Rstar = float(lines[3][9:])
    Tstar = float(lines[4][9:])
    Mdot  = float(lines[5][8:])     ## Msol/yr
    v     = float(lines[6][11:])    ## sec
    eps   = float(lines[8][19:])

    rtol = float(lines[31][7:])
    atol = float(lines[32][6:])

    return Rstar, Tstar, Mdot, v, eps, rtol, atol

