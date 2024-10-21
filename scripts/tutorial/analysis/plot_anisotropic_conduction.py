# PLOT ANISOTROPIC CONDUCTION

import os
import sys
import h5py, psutil, glob

import numpy as np
import multiprocessing as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import warnings
import pyharm

warnings.filterwarnings("ignore")

mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['savefig.dpi'] = 120
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.figsize'] = (7,7)
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


# Function to overlay field lines
# Argument must be axes object, dump, and 'nlines' -> a parameter to account for density of field lines
def plotting_bfield_lines(ax, dump, nlines=20):
  B1 = np.squeeze(dump['B1'])
  B2 = np.squeeze(dump['B2'])
  AJ_phi = np.zeros([dump['n1'], dump['n2']])
  for j in range(dump['n2']):
    for i in range(dump['n1']):
      AJ_phi[dump['n1']-1-i,j] = (np.trapz(B2[:i,j], dx=dump['dx1']) - np.trapz(B1[i,:j], dx=dump['dx2']))
  AJ_phi -= AJ_phi.min()
  levels  = np.linspace(0, AJ_phi.max(), nlines)
  ax.contour(np.squeeze(dump['X1']), np.squeeze(dump['X2']), AJ_phi, levels=levels, colors='k')


# Plot
def plot(dumpno):
  print("Plotting dump {0:04d}".format(dumpno))

  dump = pyharm.load_dump(os.path.join(dumpsdir, 'anisotropic_conduction.out0.{:05d}.phdf'.format(dumpno)))

  fig = plt.figure()
  nrows = 2
  ncols = 1
  heights = [1,16]
  gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, height_ratios=heights, figure=fig)

  t = "{:.1f}".format(dump['t'])

  ax0 = fig.add_subplot(gs[0,:])
  ax0.annotate('t= '+str(t), xy=(0.5,0.5), xycoords='axes fraction', va='center', ha='center', fontsize='xx-large')
  ax0.axis("off")

  ax1 = fig.add_subplot(gs[1,0])
  temp_plot = ax1.pcolormesh(np.squeeze(dump['X1']), np.squeeze(dump['X2']), np.squeeze(dump['Theta']),\
    cmap = 'viridis', shading='gouraud')
  plotting_bfield_lines(ax1, dump, nlines=20)
  ax1.set_xlim(0,1)
  ax1.set_ylim(0,1)
  ax1.set_xticks([0,0.25,0.5,0.75,1])
  ax1.set_xticklabels([0,0.25,0.5,0.75,1])
  ax1.set_yticks([0,0.25,0.5,0.75,1])
  ax1.set_yticklabels([0,0.25,0.5,0.75,1])
  ax1.set_xlabel('$x$')
  ax1.set_ylabel('$y$')
  ax1.set_aspect('equal')
  divider = make_axes_locatable(ax1)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  cbar = plt.colorbar(temp_plot, cax=cax)
  cbar.set_label('P$_{g}/\\rho$', rotation=90)

  plt.savefig(os.path.join(plotsdir, 'temperature_plot_{:04d}.png'.format(dumpno)), bbox_inches='tight')
  plt.close()


if __name__=='__main__':
    dstart = int(sorted(glob.glob(os.path.join(dumpsdir,'anisotropic_conduction.out0*.phdf')))[0][-9:-5])
    dend = int(sorted(glob.glob(os.path.join(dumpsdir,'anisotropic_conduction.out0*.phdf')))[-2][-9:-5])
    dlist = range(dstart, dend+1)

    nthreads = calc_threads()
    run_parallel(plot, dlist, nthreads)