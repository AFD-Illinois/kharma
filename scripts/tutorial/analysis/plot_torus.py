# FOR 2D RUNS!!
# COMPUTES HORIZON-PENETRATING FLUXES, PLOTS SLICES OF RHO, UU, Q, DP AND TIME SERIES OF FLUXES

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

SMALL = 1.e-20

mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['savefig.dpi'] = 120
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = (4,8)
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['text.usetex'] = False
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams["font.serif"] = 'cmr10',
mpl.rcParams["font.monospace"] = 'Computer Modern Typewriter'
mpl.rcParams["mathtext.fontset"]= 'cm'
mpl.rcParams['axes.unicode_minus'] = False


# paths
dumpsdir = 'dumps_kharma'
plotsdir = 'plots'
if not os.path.exists(plotsdir):
  os.makedirs(plotsdir)


# calculate number of worker processes based on cpus
def calc_threads(pad=0.8):
  Nthreads = int(psutil.cpu_count(logical=False)*pad)
  return Nthreads


# function to parallelize plotting
def run_parallel(function, dlist,	nthreads):
	pool = mp.Pool(nthreads)
	pool.map_async(function, dlist).get(720000)
	pool.close()
	pool.join()


# Take poloidal slice of 2D array
def xz_slice(var, dump, patch_pole=False):
  xz_var = np.zeros((dump['n1'], dump['n2']))
  for i in range(dump['n1']):
    xz_var[i,:] = var[dump['n1']-1-i,:]
  if patch_pole:
    xz_var[:,0] = xz_var[:,-1] = 0
  return xz_var


# Plot
def plot(dumpno):
    print("Plotting dump {0:04d}".format(dumpno))

    # load dump
    dump = pyharm.load_dump(os.path.join(dumpsdir, 'torus.out0.{:05d}.phdf'.format(dumpno)))

    bsq = np.maximum(dump['bsq'], SMALL)
  
    x_slice   = xz_slice(np.squeeze(dump['x']), dump, patch_pole=True)
    z_slice   = xz_slice(np.squeeze(dump['z']), dump)
    rho_slice = xz_slice(np.squeeze(dump['rho']), dump)

    fig = plt.figure()
    nrows = 2
    ncols = 1
    heights = [1,16]
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, height_ratios=heights, figure=fig)

    t = "{:d}".format(int(dump['t']))

    ax0 = fig.add_subplot(gs[0,0])
    ax0.annotate('t= '+str(t)+' GM/c$^3$', xy=(0.5,0.5), xycoords='axes fraction', va='center', ha='center', fontsize='xx-large')
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[1,0])
    rho_plot = ax1.pcolormesh(x_slice, z_slice, np.log10(rho_slice), cmap = 'turbo', vmin=-5, vmax=0, shading='gouraud')
    ax1.set_xlim(-25,0)
    ax1.set_ylim(-25,25)
    ax1.set_xticks([-25,-15,-5,0])
    ax1.set_xticklabels([-25,-15,-5,0])
    ax1.set_yticks([-25,-15,-5,5,15,25])
    ax1.set_yticklabels([-25,-15,-5,5,15,25])
    ax1.set_xlabel('$x (GM/c^2)$')
    ax1.set_ylabel('$z (GM/c^2)$')
    ax1.set_title('Log$_{10}(\\rho)$')
    circle = plt.Circle((0,0), dump['r_eh'], color='silver')
    ax1.add_artist(circle)
    ax1.set_aspect('equal')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(rho_plot, cax=cax, ticks = [0, -1, -2, -3, -4, -5])
    cbar.ax.set_yticklabels([0, -1, -2, -3, -4, -5])

    plt.savefig(os.path.join(plotsdir, 'torus2d_plot_{:04d}.png'.format(dumpno)), bbox_inches='tight')
    plt.close()


if __name__=='__main__':
    dstart = int(sorted(glob.glob(os.path.join(dumpsdir,'torus.out0*.phdf')))[0][-9:-5])
    dend = int(sorted(glob.glob(os.path.join(dumpsdir,'torus.out0*.phdf')))[-2][-9:-5])
    dlist = range(dstart, dend+1)

    nthreads = calc_threads()
    run_parallel(plot, dlist, nthreads)