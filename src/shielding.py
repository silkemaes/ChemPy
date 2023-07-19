import numpy as np
from pathlib import Path

## retrieve shielding rate, given N_CO & N_H2
##      naming files:
##          CO: COshield.H2velocity[km/s].H2temp[K].13C/12C-ratio.dat
##          N2: N2shield.H2velocity[km/s].1eH2temp[K].N(H)[cm^-2].dat

def read_shielding(loc, spec):
    '''
    Read in shielding rates from tables.
    '''
    shielding = np.loadtxt(loc, skiprows=9, dtype=np.float64)
    leg = (Path(__file__).parent / f'../shielding/{spec}/{spec}legend.txt').resolve()

    spec = np.loadtxt(leg, skiprows = 3, dtype = np.float64, usecols = (0))
    H2 = np.loadtxt(leg, skiprows = 3, dtype = np.float64, usecols = (1), max_rows=shielding.shape[0])

    return shielding, spec, H2

def find_closest_lin(list, x, spec):
    '''
    Find index of list for the value closest to x. \n
    This function is specific for a loglinear 'list', \n with the first element (index 0) diverging from this relation.
    '''

    ## Consider log space.
    list = np.log10(list)
    x    = np.log10(x)

    if spec == 'CO':
        ## Exeption if x is smaller than the first value.
        if x < list[1]:
            if x < list[1]/2:
                return 0
            else:
                return 1
        ## Exeption when x is larger than the last value.
        elif x >= list[-1]:
            return int(len(list)-1)
        ## All values in between.
        else:
            list = list[1:]
            min = np.min((list))
            max = np.max((list))

            idx = np.round((x-min)*(max-min)**(-1)*(len(list)-1))

            return int(idx+1)
    
    elif spec == 'N2':
        ## Exeption when x is larger than the last value.
        if x >= list[-1]:
            return int(len(list)-1)
        ## All values in between.
        else:
            min = np.min((list))
            max = np.max((list))

            idx = np.round((x-min)*(max-min)**(-1)*(len(list)-1))

            return int(idx)
        


def get_shield_table(T, Av, spec):
    '''
    Function to select the best corresponding shielding table.
    '''
    temp = select_shield_temp(T, spec)

    if spec == 'N2':
        N_H = select_shield_NH(Av)

        return temp, N_H

    elif spec == 'CO':
        return temp

def select_shield_temp(T, spec):
    '''
    Function to select the best corresponding temperature to select the sheilding table.
    '''
    T_list_CO = np.array([5,20,50,100])
    T_list_N2 = np.array([10,30,50,100,1000])

    if spec == 'CO':
        list = T_list_CO
    elif spec == 'N2':
        list= T_list_N2

    idx = find_closest(list, T)
    temp_select = list[idx]

    return temp_select

def select_shield_NH(Av):
    '''
    Function to select the best corresponding N_H (H column density) to select the sheilding table. \n
    Only in the case of N2 shielding.
    '''
    N_H2 = Av * 1.87e21
    N_H = 2* N_H2

    NH_list = np.array([1e14,1e20,1e22])

    idx = find_closest(NH_list, N_H)
    N_H_select = NH_list[idx]

    return int(np.log10(N_H_select))

def select_shield_v(v):
    '''
    Function to select the best corresponding v (velocity) to select the sheilding table. \n
    '''
    v_list = np.array([3,11])

    idx = find_closest(v_list, v)
    v_select = v_list[idx]

    return v_select


def find_closest(list, target):
    '''
    Helper function to find the element in the list closest to the target. \n
    Returns the corresponding index.
    '''
    idx = 0
    for i in range(len(list)-1):
        if np.abs(list[i+1]-target) <= np.abs(list[i]-target):
            idx = i+1

    return idx

def retrieve_rate(n_i, Av, T, vexp, C13C12, spec):
    '''
    Retrieve the shielding rate, corresponding best to the current modelling input parameters. \n
    Input: \n
        - N         = target column density of the specific species (CO or N2)\n
        - N_H2      = target column density of H2 \n
        - shielding = shielding table in 2D-np.array \n
        - spec      = list with column densities from specie \n
        - H2        = list with column densities from H2
    '''

    dir_shielding = 'shielding/'
    v = select_shield_v(vexp)
    if spec == 'CO':
        temp = get_shield_table(T, Av, spec)
        loc = spec+'shield.'+str(v)+'.'+str(temp)+'.'+str(C13C12)
    if spec == 'N2':
        temp, N_H = get_shield_table(T, Av, spec)
        loc = spec+'shield.'+str(v)+'.'+str(temp)+'.'+str(N_H)

    shielding, lgd_spec, lgd_H2 = read_shielding((Path(__file__).parent / f'../{dir_shielding}{spec}/{loc}.dat').resolve(), spec)

    N_H2 = Av * 1.87e21
    N = N_H2 * n_i[spec] 

    idx = find_closest_lin(lgd_spec, N, spec)
    idx_H2 = find_closest_lin(lgd_H2, N_H2, spec)

    shieldrate = shielding[idx_H2, idx]

    return shieldrate