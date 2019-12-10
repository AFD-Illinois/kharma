################################################################################
#                                                                              #
#  GENERATE MOVIES COMPARING 2 SIMULATIONS' OUTPUT                             #
#                                                                              #
################################################################################

# Local
import hdf5_to_dict as io
import plot as bplt
from analysis_fns import *
import util

# System
import sys; sys.dont_write_bytecode = True
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
import pickle

# Movie size in ~inches. Keep 16/9 for standard-size movies
FIGX = 10
FIGY = 8
#FIGY = FIGX*9/16

def plot(n):
  imname = os.path.join(frame_dir, 'frame_%08d.png' % n)
  tdump = io.get_dump_time(files1[n])
  if (tstart is not None and tdump < tstart) or (tend is not None and tdump > tend):
    return

  # Don't calculate b/ucon/cov/e- stuff unless we need it below
  dump1 = io.load_dump(files1[n], hdr1, geom1, derived_vars = False, extras = False)
  dump2 = io.load_dump(files2[n], hdr2, geom2, derived_vars = False, extras = False)

  fig = plt.figure(figsize=(FIGX, FIGY))

  # Keep same parameters betwen plots, even of SANE/MAD
  rho_l, rho_h = -3, 2
  window = [-20,20,-20,20]
  nlines1 = 20
  nlines2 = 5
  
  # But BZ stuff is done individually
  if hdr1['r_out'] < 100:
    iBZ1 = i_of(geom1,40) # most SANEs
    rBZ1 = 40
  else:
    iBZ1 = i_of(geom1,100) # most MADs
    rBZ1 = 100
  if hdr2['r_out'] < 100:
    iBZ2 = i_of(geom2,40)
    rBZ2 = 40
  else:
    iBZ2 = i_of(geom2,100)
    rBZ2 = 100

  if movie_type == "simplest":
    # Simplest movie: just RHO
    gs = gridspec.GridSpec(1, 2)
    ax_slc = [plt.subplot(gs[0]), plt.subplot(gs[1])]
    bplt.plot_xz(ax_slc[0], geom, np.log10(dump1['RHO']), label=r"$\log_{10}(\rho)$, MAD",
                 ylabel=False, vmin=rho_l, vmax=rho_h, window=window, half_cut=True, cmap='jet')
    bplt.overlay_field(ax_slc[0], geom, dump1, nlines1)
    bplt.plot_xz(ax_slc[1], geom, np.log10(dump2['RHO']), label=r"$\log_{10}(\rho)$, SANE",
                 ylabel=False, vmin=rho_l, vmax=rho_h, window=window, half_cut=True, cmap='jet')
    bplt.overlay_field(ax_slc[1], geom, dump2, nlines2)
  elif movie_type == "simpler":
    # Simpler movie: RHO and phi
    gs = gridspec.GridSpec(2, 2, height_ratios=[5, 1])
    ax_slc = [plt.subplot(gs[0,0]), plt.subplot(gs[0,1])]
    ax_flux = [plt.subplot(gs[1,:])]
    
    bplt.plot_xz(ax_slc[0], geom, np.log10(dump1['RHO']), label=r"$\log_{10}(\rho)$, MAD",
                 ylabel=False, vmin=rho_l, vmax=rho_h, window=window, cmap='jet')
    bplt.overlay_field(ax_slc[0], geom, dump1, nlines1)
    bplt.plot_xz(ax_slc[1], geom, np.log10(dump2['RHO']), label=r"$\log_{10}(\rho)$, SANE",
                 ylabel=False, vmin=rho_l, vmax=rho_h, window=window, cmap='jet')
    bplt.overlay_field(ax_slc[1], geom, dump2, nlines2)

    # This is way too custom
    ax = ax_flux[0]; ylim=[0,80]
    slc1 = np.where((diag1['phi'] > ylim[0]) & (diag1['phi'] < ylim[1]))
    slc2 = np.where((diag2['phi'] > ylim[0]) & (diag2['phi'] < ylim[1]))
    ax.plot(diag1['t'][slc1], diag1['phi'][slc1], 'r', label="MAD")
    ax.plot(diag2['t'][slc2], diag2['phi'][slc2], 'b', label="SANE")
    ax.set_xlim([diag1['t'][0], diag1['t'][-1]])
    ax.axvline(dump1['t'], color='r')
    ax.set_ylim(ylim)
    ax.set_ylabel(r"$\phi_{BH}$")
    ax.legend(loc=2)
  elif movie_type == "rho_cap":
    axes = [plt.subplot(2,3,i) for i in range(1,7)]
    
    bplt.plot_slices(axes[0], axes[1], geom1, dump1, np.log10(dump1['RHO']),
                   label=r"$\log_{10}(\rho) (1)$", vmin=-3, vmax=2, cmap='jet')
    bplt.overlay_contours(axes[0], geom1, geom1['r'], [rBZ1], color='k')
    bplt.plot_thphi(axes[2], geom1, np.log10(dump1['RHO'][iBZ1,:,:]), iBZ1, vmin=-4, vmax=1,
                    label=r"$\log_{10}(\rho)$ $\theta-\phi$ slice r="+str(rBZ1)+" (1)")
    
    bplt.plot_slices(axes[3], axes[4], geom2, dump2, np.log10(dump2['RHO']),
                   label=r"$\log_{10}(\rho) (2)$", vmin=-3, vmax=2, cmap='jet')
    bplt.overlay_contours(axes[3], geom2, geom2['r'], [rBZ2], color='k')
    bplt.plot_thphi(axes[5], geom2, np.log10(dump2['RHO'][iBZ2,:,:]), iBZ2, vmin=-4, vmax=1,
                    label=r"$\log_{10}(\rho)$ $\theta-\phi$ slice r="+str(rBZ2)+" (2)")


  pad = 0.05
  plt.subplots_adjust(left=pad, right=1-pad, bottom=2*pad, top=1-pad)
  plt.savefig(imname, dpi=1920/FIGX)
  plt.close(fig)

if __name__ == "__main__":
  # PROCESS ARGUMENTS
  if sys.argv[1] == '-d':
    debug = True
    movie_type = sys.argv[2]
    path1 = sys.argv[3]
    path2 = sys.argv[4]
    if len(sys.argv) > 5:
      tstart = float(sys.argv[5])
    if len(sys.argv) > 6:
      tend = float(sys.argv[6])
  else:
    debug = False
    movie_type = sys.argv[1]
    path1 = sys.argv[2]
    path2 = sys.argv[3]
    if len(sys.argv) > 4:
      tstart = float(sys.argv[4])
    if len(sys.argv) > 5:
      tend = float(sys.argv[5])
  
  # LOAD FILES
  files1 = io.get_dumps_list(path1)
  files2 = io.get_dumps_list(path2)
  
  if len(files1) == 0 or len(files2) == 0:
      util.warn("INVALID PATH TO DUMP FOLDER")
      sys.exit(1)

  frame_dir = "frames_compare_"+movie_type
  util.make_dir(frame_dir)

  hdr1 = io.load_hdr(files1[0])
  hdr2 = io.load_hdr(files2[0])
  geom1 = io.load_geom(hdr1, path1)
  geom2 = io.load_geom(hdr2, path2)
  # TODO diags from post?
  # Load diagnostics from HARM itself
  diag1 = io.load_log(path1)
  diag2 = io.load_log(path2)

  nthreads = util.calc_nthreads(hdr1)
  if debug:
    for i in range(len(files1)):
      plot(i)
  else:
    util.run_parallel(plot, len(files1), nthreads)
