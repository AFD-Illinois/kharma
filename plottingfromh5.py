import numpy as np
import os, h5py, psutil, glob
import multiprocessing as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import turbo_colormap_mpl
import matplotlib as mpl
import warnings

import pyharm

mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['savefig.dpi'] = 120
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = (16,9)
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams["font.serif"] = 'cmr10',
# mpl.rcParams["font.monospace"] = 'Computer Modern Typewriter'
# mpl.rcParams["mathtext.fontset"]= 'cm'


# NOTE
# 1. Make sure number of dumps specified in 'load_fluxes' function !! 
# 2. Image will be saved wherever code is run !! 
# 3. Plot phibh and mdot seperately 

import os
import h5py
import matplotlib.pyplot as plt

import os
import h5py
import matplotlib.pyplot as plt

def load_fluxes(fluxdir,theory):
    fluxes = {'t': [], 'phibh': [], 'mdot': []}

    for dumpno in range(0, 100):  # Assuming flux files range from 0 to 2187
        print(f"Loading fluxes for theory ", theory, " Dump ", dumpno)
        hfp = h5py.File(os.path.join(fluxdir, f'flux_reduction_{dumpno}.h5'), 'r')
        fluxes['t'].append(hfp['t'][()])
        fluxes['phibh'].append(hfp['phibh'][()])
        fluxes['mdot'].append(hfp['mdot'][()])
        hfp.close()

    return fluxes

if __name__ == '__main__':

    # Directories of fluxes 
    fluxdir_gr = '/scratch/bbgv/smajumdar/FLUXES_TEST/gr'  
    fluxdir_dcs = '/scratch/bbgv/smajumdar/FLUXES_TEST/dcs'  
    fluxdir_edgb = '/scratch/bbgv/smajumdar/FLUXES_TEST/edgb' 

    fluxes_gr = load_fluxes(fluxdir_gr,'GR')
    fluxes_dcs = load_fluxes(fluxdir_dcs,'DCS')
    fluxes_edgb = load_fluxes(fluxdir_edgb,'EDGB')

    # Plotting mdot OR phibh
    plt.figure(figsize=(16, 6))
    
    plt.plot(fluxes_gr['t'], fluxes_gr['phibh'], label='GR',color='gray')
    plt.plot(fluxes_dcs['t'], fluxes_dcs['phibh'], label='dCS', color='maroon')
    plt.plot(fluxes_edgb['t'], fluxes_edgb['phibh'], label='EdGB',color='olive')

    plt.xlabel('$t (GM/c^3)$')
    plt.ylabel('Magnetic Flux')
    plt.grid(True)
    plt.title('Time Series of Magnetic Flux for GR, dCS, and EdGB')

    
    plt.legend()

    # Change name based on variable! 
    plt.savefig('MagneticFlux.png')
