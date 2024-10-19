### Plot density and norm of 4-current ###

import numpy as np
import sys, glob, psutil, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprocessing as mp
import pyharm
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['savefig.dpi'] = 120
mpl.rcParams['figure.autolayout'] = True
# mpl.rcParams['figure.figsize'] = (8,4)
mpl.rcParams['figure.figsize'] = (6,6)
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
 

# plot function (parallelized)
def plot(dumpno):
    print("Plotting dump {:04d}".format(dumpno))
    # plotting parameters
    vmin_rho = 0; vmax_rho = 1
    vmin_jsq = -3; vmax_jsq = 0
    cmap_rho = 'turbo'
    cmap_jsq = 'plasma'
    shading = 'gouraud'

    # load dump
    dump = pyharm.load_dump(os.path.join(dumpsdir, 'orszag_tang.out0.{:05d}.phdf'.format(dumpno)))

    t = '{:.1f}'.format(dump['t'])

	# plot	
    fig = plt.figure()
    nrows = 2
    # ncols = 2
    ncols = 1
    heights = [1,10]
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols, height_ratios=heights, figure=fig)

    ax0 = fig.add_subplot(gs[0,:])
    ax0.annotate('t= '+str(t), xy=(0.5,0.5), xycoords='axes fraction', va='center', ha='center', fontsize = 'x-large')
    ax0.axis('off')

    # ax1 = fig.add_subplot(gs[1,0])
    # rho_plot = ax1.pcolormesh(np.squeeze(dump['X1']), np.squeeze(dump['X2']), np.log10(np.squeeze(dump['rho'])), \
    #             cmap=cmap_rho, vmin=vmin_rho, vmax=vmax_rho, shading=shading)
    # ax1.set_xlabel('$x$')
    # ax1.set_ylabel('$y$')
    # ax1.set_title('Log$_{10}(\\rho$)')
    # ax1.set_aspect('equal')
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(rho_plot, cax=cax)
    
    # ax2 = fig.add_subplot(gs[1,1])
    ax2 = fig.add_subplot(gs[1,0])
    jsq_plot = ax2.pcolormesh(np.squeeze(dump['X1']), np.squeeze(dump['X2']), np.log10(np.squeeze(dump['jsq'])), \
                cmap=cmap_jsq, vmin=vmin_jsq, vmax=vmax_jsq, shading=shading)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')
    ax2.set_title('Log$_{10}(j^{2}$)')
    ax2.set_aspect('equal')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(jsq_plot, cax=cax)

    plt.savefig(os.path.join(plotsdir,'orszag_tang_{0:04d}.png'.format(dumpno)), bbox_inches='tight')
    plt.close()

if __name__=="__main__":
    dstart = int(sorted(glob.glob(os.path.join(dumpsdir,'orszag_tang.out0*.phdf')))[0][-9:-5])
    dend = int(sorted(glob.glob(os.path.join(dumpsdir,'orszag_tang.out0*.phdf')))[-2][-9:-5])
    dlist = range(dstart,dend+1)

    nthreads = calc_threads()
    run_parallel(plot,dlist,nthreads)