from astropy import units   as units
import numpy                as np
import sys
from scipy.interpolate  import interp1d
import os

import json


sys.path.append('/STER/silkem/ChemTorch/')

from src.solve_n_save       import solve
from src.input              import density
import src.rates            as rates


def makeOutputDir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

rate = 16

outloc = '/STER/silkem/ChemTorch/out/'
samploc = '/STER/silkem/ChemTorch/sampling/'
dirname = 'torchode-test'
# dirname = 'new'


## Ranges from PHANTOM models
ρ_min = min(np.load(samploc+'drho_range.npy'))
ρ_max = max(np.load(samploc+'drho_range.npy'))
T_min = min(np.load(samploc+'dT_range.npy'))
T_max = max(np.load(samploc+'dT_range.npy'))
δ_min = 1.e-6
δ_max = 1
Av_min = 0
Av_max = 6
dt_min = min(np.load(samploc+'dtime_range.npy'))
dt_max = max(np.load(samploc+'dtime_range.npy'))

makeOutputDir(outloc+dirname+'/')


nstep = 512

### !!! HIER VERONDERSTELLEN WE ONGECORRELEERDE PARAMETERS


def fdens(x):
	rho_func = np.load(samploc+'drho.npy') 
	return rho_func

def ftemp(x):
	T_func = np.load(samploc+'dT.npy') 
	return T_func

def fdelta(x):
	return np.ones_like(x)

def fAv(x):
	return np.ones_like(x)

# def fdt(x):
# 	return x**3./(-1.+np.exp(-x+3.8001))

def fdt(x):
	time_func = np.load(samploc+'dtime.npy') 
	return time_func

## cummulative sum 
## generate random numbers between [0,1)
## Define a function to return N samples
def genSamples(xmin, xmax, nstep, N, f):
	if f == fdelta:
		xbin = np.logspace(np.log10(xmin), np.log10(xmax), nstep)
	else:
		xbin = np.linspace(xmin, xmax, nstep)
	# print(xmin,xmax)
	ycum = np.cumsum(f(xbin))
	# plt.plot(ycum)
	if f == fdelta:
		xbin = np.logspace(np.log10(xmin), np.log10(xmax), len(ycum))
	else:
		xbin = np.linspace(xmin, xmax, len(ycum))
	u = np.random.uniform(ycum.min(), ycum.max(), int(N))
	## take the inverse of cumm. function
	func_interp = interp1d(ycum, xbin)
	samples = func_interp(u)
	return samples
	

def calc_next(f, param_i, min, max, nstep):
	N = 1
	fact = 1
	# if param_i > 1e7:
	# fact = 10
	ε = fact*genSamples(min, max, nstep, N, f)
	# print(ε)
	param_next = (ε + 1)*param_i
	return param_next[0]

def next_input_idv(ρ, T, δ, Av):
	ρ_next  = calc_next(fdens , ρ , ρ_min , ρ_max , nstep)
	T_next  = calc_next(ftemp , T , T_min , T_max , nstep)
	δ_next  = calc_next(fdelta, δ , δ_min , δ_max , nstep)
	Av_next = calc_next(fAv   , Av, Av_min, Av_max, nstep)
	return ρ_next, T_next, δ_next, Av_next

def next_input(input):
	ρ = input[0]
	T = input[1]
	ρ_next  = calc_next(fdens , ρ , ρ_min , ρ_max , nstep)
	T_next  = calc_next(ftemp , T , T_min , T_max , nstep)
	δ_next  = genSamples(δ_min , δ_max , nstep, 1, fdelta)[0]
	Av_next = genSamples(Av_min, Av_max, nstep, 1, fAv)[0]
	return [ρ_next, T_next, δ_next, Av_next]

def get_dt():
	dt = genSamples(dt_min, dt_max, nstep, 1, fdt)[0] 
	return dt

def get_temp(T, eps, r):
    R_star = 1.0e14            ## cm
    temp = T*(r/R_star)**-eps
    return temp



Mdot = 1e-7
v = 10
T_star = 2500
eps = 0.4
r = np.array(np.logspace(14,18, 100))
dens = density(Mdot, v,r )
temp = get_temp(T_star,eps, r)


metadata = {
	'rel_rho_min' : ρ_min,
	'rel_rho_max' : ρ_max,
	'rel_T_min' : T_min,
	'rel_T_max' : T_max,
	'delta_min' : δ_min,
	'delta_max' : δ_max,
	'Av_min'    : Av_min,
	'Av_max'    : Av_max,
	'dt_min'	: dt_min,
	'dt_max'    : dt_max,
	'Mdot' 		: Mdot,
	'v'			: v,
	'T_star'	: T_star,
	'eps'		: eps,
	'r_range'	: [14,18],
	'solvertype': ''
}

# print(np.log10(dens[0]))
# print(temp[0])
# xxx

json_object = json.dumps(metadata, indent=4)
with open(outloc+dirname+"/meta.json", "w") as outfile:
    outfile.write(json_object)

i=0
# for i in range(len(dens)):
chemtype = 'C'

## set initial conditions
n, nconsv_tot, specs, nshield_i = rates.initialise_abs(chemtype, rate)     # nconsv_tot = TOTAL in fortran code

δi  = 1.e-1
Avi = -np.log(1.e-3)
input = [dens[i],temp[i],δi,Avi]
name = '' 

while input[0] > 10. and input[1] > 10.:
	# dt = get_dt()    ## sec
	dt = 5000
	if dt < 10000:
		n, name = solve(input, dt, rate, n, nshield_i, nconsv_tot, name, dirname=dirname)
		input = next_input(input)
	break
