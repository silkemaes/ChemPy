import numpy                as np
import matplotlib.pyplot    as plt
from astropy import units   as units
import datetime as dt
import sys
from pathlib            import Path
from matplotlib         import rcParams, rc
from scipy.interpolate  import interp1d
import types
# Change the matplotlib default parameters
rcParams.update({'font.size':   8})
rcParams.update({'figure.dpi': 120})

sys.path.append('/lhome/silkem/ChemTorch/ChemTorch/')
from src.read               import read_data_chemtorch
# from src.rates          import *
from src.solve_n_save       import solve_dg
from src.input              import density

rate = 16

outloc = '/lhome/silkem/ChemTorch/ChemTorch/out/'
samploc = '/lhome/silkem/ChemTorch/ChemTorch/sampling/'
dataloc = '/lhome/silkem/ChemTorch/PhantomSampling/'

## Ranges from PHANTOM models
ρ_min = min(np.load(samploc+'drho_range.npy'))
ρ_max = max(np.load(samploc+'drho_range.npy'))
T_min = min(np.load(samploc+'dT_range.npy'))
T_max = max(np.load(samploc+'dT_range.npy'))
δ_min = 1.e-6
δ_max = 1
Av_min = -1
Av_max = -np.log(δ_max)
dt_min = min(np.load(samploc+'dtime_range.npy'))
dt_max = max(np.load(samploc+'dtime_range.npy'))

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
	xbin = np.linspace(xmin, xmax, nstep)
	# print(xmin,xmax)
	ycum = np.cumsum(f(xbin))
	# plt.plot(ycum)
	xbin = np.linspace(xmin, xmax, len(ycum))
	u = np.random.uniform(ycum.min(), ycum.max(), int(N))
	## take the inverse of cumm. function
	func_interp = interp1d(ycum, xbin)
	samples = func_interp(u)
	return samples
	

def calc_next(f, param_i, min, max, nstep):
	N = 1
	ε = genSamples(min, max, nstep, N, f)
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
	δ = input[2]
	Av= input[3]
	ρ_next  = calc_next(fdens , ρ , ρ_min , ρ_max , nstep)
	T_next  = calc_next(ftemp , T , T_min , T_max , nstep)
	δ_next  = calc_next(fdelta, δ , δ_min , δ_max , nstep)
	Av_next = calc_next(fAv   , Av, Av_min, Av_max, nstep)
	# print(ρ_next.type)
	return [ρ_next, T_next, δ_next, Av_next]

def get_dt():
	return genSamples(dt_min, dt_max, nstep, 1, fdt)[0] 




# nsamples = 1.e4
# nbins = 10000

# dens_samples = genSamples(ρ_min, ρ_max, nstep, nsamples, fdens)
# temp_samples = genSamples(T_min, T_max, nstep, nsamples, ftemp)
# delt_samples = genSamples(δ_min, δ_max, nstep, nsamples, fdelta)
# Av_samples = genSamples(Av_min, Av_max, nstep, nsamples, fAv)
# dt_samples = genSamples(dt_min, dt_max, nstep, nsamples, fdt)

# fig = plt.figure(figsize=(13,7))

# ax1 = plt.subplot(231)
# ax2 = plt.subplot(232)
# ax3 = plt.subplot(233)
# ax4 = plt.subplot(234)
# ax5 = plt.subplot(235)

# ax1.hist(delt_samples,bins=nbins, density=True, color='y', alpha=0.7, label = 'delta')
# ax2.hist(Av_samples  ,bins=nbins, density=True, color='g', alpha=0.7, label = 'Av')
# ax3.hist(dens_samples,bins=nbins, density=True, color='r', alpha=0.7, label = 'density')
# ax4.hist(temp_samples,bins=nbins, density=True, color='b', alpha=0.7, label = 'temp')
# ax5.hist(dt_samples  ,bins=nbins, density=True, color='grey', alpha=0.7, label = 'dt')

# fig.legend(loc = 'upper left')
# plt.show()