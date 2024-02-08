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

warnings.filterwarnings("ignore")

# SMALL = 1.e-20

params = {}
fluxes = {}

# Parallel Computiing 
def calc_threads(pad=0.4):
  Nthreads = int(psutil.cpu_count(logical=False)*pad)
  return Nthreads


def run_parallel(function, args_list, nthreads):
  pool = mp.Pool(nthreads)
  pool.starmap_async(function, args_list).get(720000)
  pool.close()
  pool.join()


# FUNCTIONS 
# Compute sum over a full shell
def shell_sum(var, dump, rind):
  return np.sum(np.sum(var[rind,:,:] * dump['gdet'][rind,:,:], axis=1) * dump['dx3'], axis=0) * dump['dx2']


# Time, Accretion rate and Magnetic Flux calculation 
def calc_fluxes(params, dumpno, theory, fluxes_dict):
    print(f"Computing fluxes for dump {dumpno:04d} in {theory} theory")

    # Create theory-specific directory if it doesn't exist
    theory_dir = os.path.join(params['fluxdir'], theory)
    if not os.path.exists(theory_dir):
        os.makedirs(theory_dir)

    # Load data based on the theory (GR, dCS, or EdGB)
    dump = pyharm.load_dump(os.path.join(params[f'dumpsdir_{theory.lower()}'], f'torus.out0.{dumpno:05d}.phdf'))
    reh_ind = np.argmin(np.fabs(np.squeeze(dump['r'][:, 0, 0]) - dump['r_eh']))

    mdot = -shell_sum(np.squeeze(dump['rho'] * dump['ucon'][1, Ellipsis]), dump, reh_ind)
    phibh = 0.5 * shell_sum(abs(np.squeeze(dump['B'][0, Ellipsis])), dump, reh_ind)
    time = dump['t']

    fluxes_dict[f't_{theory}'].append(time)
    fluxes_dict[f'mdot_{theory}'].append(mdot)
    fluxes_dict[f'phibh_{theory}'].append(phibh)

    # Save data to theory-specific directory
    hfp = h5py.File(os.path.join(theory_dir, f'flux_reduction_{dumpno}.h5'), 'w')
    hfp['t'] = dump['t']
    hfp['mdot'] = mdot
    hfp['phibh'] = phibh
    hfp.close()

if __name__ == '__main__':

    print("Starting the script...")
    # Set directories for each theory
    params['dumpsdir_gr'] = '/scratch/bbgv/smajumdar/T_runs_7000M/vanilla/dumps_kharma'
    params['dumpsdir_dcs'] = '/scratch/bbgv/smajumdar/T_runs_7000M/dcs/dumps_kharma'
    params['dumpsdir_edgb'] = '/scratch/bbgv/smajumdar/T_runs_7000M/edgb/dumps_kharma'
    params['fluxdir'] = '/scratch/bbgv/smajumdar/FLUXES_TEST'

    if not os.path.exists(params['fluxdir']):
      os.makedirs(params['fluxdir'])

    # Set the theory for which you want to analyze data (e.g., 'GR', 'dCS', or 'EdGB')
    theories = ['gr', 'dcs', 'edgb']

    fluxes = {}
    for theory in theories:
        fluxes[f't_{theory}'] = []
        fluxes[f'mdot_{theory}'] = []
        fluxes[f'phibh_{theory}'] = []

        dlist = range(0, 100)

        nthreads = calc_threads(pad=0.4)
        run_parallel(calc_fluxes, [(params, dumpno, theory, fluxes) for dumpno in dlist], nthreads)

        print("Parallel computation completed.")