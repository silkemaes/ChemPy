from astropy import units   as units
import numpy                as np
import sys
import os
import datetime         as dt
import torch
import torchode     as to


from astropy import constants   as cst
from astropy import units       as units
from astropy.constants          import M_sun
import matplotlib.pyplot        as plt

import json

## units & constants
Msun = M_sun.cgs.value      # gram
yr   = 31536000             # s
Msunyr = Msun/yr            # gram/s
cms  = 1e5


## Physical constants
kB = cst.k_B.cgs.value          ## Boltzmann constant [erg/K]
mH = cst.u.to('g').value        ## mass H atom = atomic mass unit [g]
Msun = cst.M_sun.cgs.value      ## gram
yr = units.year.to('s')         ## year in seconds
Msunyr = Msun/yr                ## units of mass-loss rate in gram/s
cms  = 1e5                      ## units of velocity in cm/s
g_to_kg = units.gram.to('kg')
cm_to_m = units.cm.to('m')

mu = 2.0 + 4.0*0.17             ## mu (average mass per H2 molecule), taking into account the abundance of He



sys.path.append('/STER/silkem/ChemTorch/')

from src.solve_n_save       import solve
import src.rates            as rates
from src.ode.acodes_torch import torchODE


def makeOutputDir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

rate = 16

## location to save benchmark
out = '/STER/silkem/ChemTorch/out/'
dirname = 'bm_torch'

## 1D chem model
outloc = '/STER/silkem/CSEchem/'
outdir = '20210521_gridC_Mdot1e-6_v15_T_eps-'
mod = 'model_2022-12-26h13-01-25'

makeOutputDir(out+dirname+'/')

## input
Mdot = 1.e-6
v = 15.
eps = 0.6
T_star = 2500
solvertype = 'torch'

## loading the physical input from the 1D model
arr = np.loadtxt(outloc+outdir+mod+'/csphyspar_smooth.out', skiprows=4, usecols=(0,1,2,3,4,11))
radius, dens, temp, Av, δ, delta_AUV = arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4], arr[:,5]
time = radius/(v* cms)
# time = 10**(np.log10(time)-np.log10(time)[0])
chemtype = 'C' 

atol = 1.e-20
rtol = 1.e-5


dt = np.zeros(len(time))
for i in range(1,len(time)):
    dt[i] = (time[i]-time[i-1])


## metadata
metadata = {'1Dmodel'   : outloc+outdir+mod,
            'Mdot'      : Mdot,
            'v'         : v,
            'eps'       : eps,
            'T_star'    : T_star,
            'solvertype': solvertype
        }

json_object = json.dumps(metadata, indent=4)
with open(out+dirname+"/meta.json", "w") as outfile:
    outfile.write(json_object)


## set initial conditions
n, nconsv_tot, specs, nshield_i = rates.initialise_abs(chemtype, rate)    
name = '' 

## build & compile torch ODE solver
# if solvertype == 'torch':
# 	odeterm = to.ODETerm(torchODE, with_args=True)
# 	step_method          = to.Dopri5(term=odeterm)
# 	step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=odeterm)
# 	adjoint              = to.AutoDiffAdjoint(step_method, step_size_controller) # type: ignore
# 	jit_solver = torch.compile(adjoint)

# if solvertype == 'scipy':
jit_solver = None



for i in range(len(dens)):
    input = [dens[i], temp[i], δ[i], Av[i]]
    n, name = solve(input, dt[i], rate, n, nshield_i, nconsv_tot, name, dirname=dirname, solvertype = solvertype, jitsolver=jit_solver, atol=atol, rtol=rtol) # type: ignore


