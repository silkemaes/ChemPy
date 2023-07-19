import numpy as np

def read_data_fortran(file_name):
    '''
    Read data text file of output abundances of fortran dc models
    '''
    ## Get abundances
    with open(file_name, 'r') as file:
        dirty = []
        proper = None
        for i, line in enumerate(file):
            try:                
                if len(line) > 1: 
                    dirty.append([float(el) for el in line.split()])
            except:
                if len(dirty) != 0:
                    max_rows = len(dirty)
                    dirty = np.array(dirty)[:,1:]
                    if proper is None:
                        proper = dirty
                    else:
                        proper = np.concatenate((proper, dirty), axis = 1)
                dirty = []
        abs = proper.T

    ## get time array
    time = np.loadtxt(file_name, usecols=0, skiprows=1, max_rows=max_rows)

    return time, abs

def read_data_chemtorch(file_name):
    '''
    Read data text file of output abundances of ChemTorch models
    '''
    data = np.loadtxt(file_name,skiprows=1)
    data = data.T
    time = data[0]
    abs = data[1:-1]
    return time, abs