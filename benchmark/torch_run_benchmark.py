from astropy import units   as units
import numpy                as np
import sys
import os
from os                     import listdir
import datetime             as dt
import torch
import torchode             as to
from natsort                import natsorted

from astropy import constants   as cst
from astropy import units       as units
from astropy.constants          import M_sun
import json

## units & constants
Msun = M_sun.cgs.value      # gram
yr   = 31536000             # s
Msunyr = Msun/yr            # gram/s


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
from src.ode.acodes_torch   import torchODE
import modclass


def makeOutputDir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

rate = 16

## -----------------------------   SETUP

## location to save benchmark
out = '/STER/silkem/ChemTorch/out/'
dirname = 'bm_C_Mdot1e-5_v20'

## 1D chem model
outloc = '/STER/silkem/CSEchem/'

# outdir = '20210518_gridC_Mdot1e-8_v2-5_T_eps'
# mod = 'model_2022-12-24h23-19-06'

# outdir = '20210521_gridC_Mdot1e-6_v15_T_eps'
# mod = 'model_2022-12-26h13-01-25'

outdir = '20210527_gridC_Mdot1e-5_v20_T_eps'
mod = 'model_2022-12-27h14-01-50'

solvertype = 'scipy'
chemtype = 'C' 
## ODE solver set
atol = 1.e-20
rtol = 1.e-5
## --------------------------------------


makeOutputDir(out+dirname+'/')

## loading the physical input from the 1D model
CSEmodel = modclass.CSEmod(loc = 'STER', dir = outdir, modelname = mod)

## input
Mdot   = CSEmodel.Mdot
v      = CSEmodel.v
eps    = CSEmodel.eps
T_star = CSEmodel.Tstar

## parametrised inputs
dens = CSEmodel.dens
temp = CSEmodel.temp
δ    = CSEmodel.delta
Av   = CSEmodel.Av
time = CSEmodel.time



## Remesh for the torchode benchmark
if solvertype == 'torch':
    print('yes torch')
    t = np.linspace(min(time), 1.e9, 5000)

    dens = np.interp(t, time, dens)
    temp = np.interp(t, time, temp)
    Av   = np.interp(t, time, Av  )
    δ    = np.interp(t, time, δ   )



if solvertype == 'torch':
    print('yes torch')
    time = t


dt = np.zeros(len(time))
for i in range(1,len(time)):
    dt[i-1] = (time[i]-time[i-1])


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
    print('\n>> Write meta file in '+out+dirname+'/meta.json...')
    outfile.write(json_object)


## set initial conditions
n, nconsv_tot, specs, nshield_i = rates.initialise_abs(chemtype, rate)    

name = ''


## build & compile torch ODE solver
if solvertype == 'torch':
	odeterm = to.ODETerm(torchODE, with_args=True) # type: ignore
	step_method          = to.Dopri5(term=odeterm)
	step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=odeterm)
	adjoint              = to.AutoDiffAdjoint(step_method, step_size_controller) # type: ignore
	jit_solver = torch.compile(adjoint)

if solvertype == 'scipy':
    jit_solver = None


# print(dens[0], temp[0], δ[0], Av[0])

start_outpath = '/STER/silkem/ChemTorch/out/'
## scipy implementation
start_dirname = 'bm_C_Mdot1e-8_v2-5'

## load 0D model dirs
dirs = listdir(start_outpath+start_dirname+'/')
dirs.remove('meta.json')
dirs = natsorted(dirs)

j = 14
chem = modclass.ChemTorchMod(start_dirname, dirs[j])
start_input = chem.p
start_abs   = chem.n.T[-1]

n = start_abs


## run the models
for i in range(0,len(dens)-1):
    input = [dens[i], temp[i], δ[i], Av[i]]
    n, name = solve(input, dt[i], rate, n, nshield_i, nconsv_tot, name, dirname=dirname, solvertype = solvertype,jitsolver=jit_solver, atol=atol, rtol=rtol) # type: ignore
	

