from astropy import units   as units
import numpy                as np
import sys
import os
import datetime         as dt

from astropy import constants   as cst
from astropy import units       as units
from astropy.constants          import M_sun

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

from src.solve_n_save       import solve_dg
from src.input              import density
import src.rates            as rates


def makeOutputDir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

rate = 16

## location to save benchmark
out = '/STER/silkem/ChemTorch/benchmark/'
dirname = 'bm'+str(dt.datetime.now())

## 1D chem model
outloc = '/lhome/silkem/CHEM/Output_nov2022/'
outdir = '20210521_gridC_Mdot1e-6_v15_T_eps/'
mod = 'model_2022-12-26h13-01-25'



## input
Mdot = 1.e-6
v = 15.
eps = 0.6
T_star = 2500


arr = np.loadtxt(outloc+outdir+'models/'+mod+'/csphyspar_smooth.out', skiprows=4, usecols=(0,1,2,3,4,11))
radius, n_H, temp, AUV, RAD, delta_AUV = arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4], arr[:,5]
time = radius/(v* cms)
chemtype = 'C' 


metadata = {'1Dmodel' : outloc+outdir+mod,
        'dens' : ,
        }


json_object = json.dumps(metadata, indent=4)
with open(out+dirname+"/meta.json", "w") as outfile:
    outfile.write(json_object)