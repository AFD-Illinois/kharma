#!/usr/bin/env python3

import os, sys
import pickle
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import util
import hdf5_to_dict as io
import plot as bplt
from analysis_fns import *

from defs import Met, Loci
from coordinates import dxdX_to_KS, dxdX_KS_to

FIGX=15
FIGY=15

# Decide where to measure fluxes
def i_of(geom, rcoord):
  i = 0
  while geom['r'][i,geom['n2']//2,0] < rcoord:
    i += 1
  i -= 1
  return i

def cut_pos(var, cut):
  if var.ndim > 2:
    var_X2 = bplt.flatten_xz(var, average=True)[iBZ,:hdr['n2']//2]
  else:
    var_X2 = var
  var_cut = np.where( np.logical_or(
    np.logical_and(var_X2[:-1] > cut, var_X2[1:] < cut),
    np.logical_and(var_X2[:-1] < cut, var_X2[1:] > cut)))
  return var_cut

def overlay_thphi_contours(ax, geom, avg, legend=False):
  if geom['r_out'] < 100:
    iBZ = i_of(geom, 40)
  else:
    iBZ = i_of(geom, 100)
  max_th = geom['n2']//2
  x = bplt.loop_phi(geom['x'][iBZ,:max_th,:])
  y = bplt.loop_phi(geom['y'][iBZ,:max_th,:])
  prep = lambda var : bplt.loop_phi(var[:max_th,:])
  cntrs = []
  cntrs.append(ax.contour(x,y, prep(geom['th'][iBZ,:,:]), [1.0], colors='k'))
  cntrs.append(ax.contour(x,y, prep(avg['betagamma_100_thphi']), [1.0], colors='k'))
  cntrs.append(ax.contour(x,y, prep(avg['bsq_100_thphi']/avg['rho_100_thphi']), [1.0], colors='xkcd:green'))
  cntrs.append(ax.contour(x,y, prep(avg['FE_100_thphi']), [0.0], colors='xkcd:pink'))
  cntrs.append(ax.contour(x,y, prep(avg['Be_nob_100_thphi']), [0.02], colors='xkcd:red'))
  cntrs.append(ax.contour(x,y, prep(avg['mu_100_thphi']), [2.0], colors='xkcd:blue'))
  clegends = [cnt.legend_elements()[0][0] for cnt in cntrs]
  if legend: ax.legend(clegends, [r"$\theta$ = 1", r"$\beta\gamma$ = 1", r"$\sigma$ = 1", r"FE = 0", r"Be = 0.02", r"$\mu$ = 2"])

def overlay_rth_contours(ax, geom, avg, legend=False):
  cntrs = []
  cntrs.append(bplt.overlay_contours(ax, geom, geom['th'][:,:,0], [1.0, np.pi-1.0], color='k'))
  cntrs.append(bplt.overlay_contours(ax, geom, avg['betagamma_rth'], [1.0], color='k'))
  cntrs.append(bplt.overlay_contours(ax, geom, avg['bsq_rth']/avg['rho_rth'], [1.0], color='xkcd:green'))
  cntrs.append(bplt.overlay_contours(ax, geom, avg['FE_rth'], [0.0], color='xkcd:pink'))
  cntrs.append(bplt.overlay_contours(ax, geom, avg['Be_nob_rth'], [0.02], color='xkcd:red'))
  cntrs.append(bplt.overlay_contours(ax, geom, avg['mu_rth'], [2.0], color='xkcd:blue'))
  clegends = [cnt.legend_elements()[0][0] for cnt in cntrs]
  if legend: ax.legend(clegends, [r"$\theta$ = 1", r"$\beta\gamma$ = 1", r"$\sigma$ = 1", r"FE = 0", r"Be = 0.02", r"$\mu$ = 2"], loc='upper right')

def overlay_th_contours(ax, avg):
  th_cut1 = cut_pos(avg['th100'][:hdr['n2']//2], 1.0)
  bg_cut1 = cut_pos(np.mean(avg['betagamma_100_thphi'], axis=-1)[:hdr['n2']//2], 1.0)
  sigma_cut1 = cut_pos(np.mean(avg['bsq_100_thphi']/avg['rho_100_thphi'],axis=-1)[:hdr['n2']//2], 1.0)
  fe_cut0 = cut_pos(np.mean(avg['FE_100_thphi'],axis=-1)[:hdr['n2']//2], 0.0)
  be_nob0_cut = cut_pos(avg['Be_nob_100_th'][:hdr['n2']//2]/hdr['n3'], 0.02)
  mu_cut2 = cut_pos(avg['mu_100_thphi'][:hdr['n2']//2]/hdr['n3'], 2.0)
  
  ylim = ax.get_ylim()
  ax.vlines(avg['th100'][th_cut1], ylim[0], ylim[1], colors='k', label=r"$\theta$ = 1")
  ax.vlines(avg['th100'][bg_cut1], ylim[0], ylim[1], colors='k', label=r"$\beta\gamma$ = 1")
  ax.vlines(avg['th100'][sigma_cut1], ylim[0], ylim[1], colors='xkcd:green', label=r"$\sigma$ = 1")
  ax.vlines(avg['th100'][fe_cut0], ylim[0], ylim[1], colors='xkcd:pink', label=r"FE = 0")
  ax.vlines(avg['th100'][be_nob0_cut], ylim[0], ylim[1], colors='xkcd:red', label=r"Be = 0.02")
  #ax.vlines(avg['th100'][mu_cut2], ylim[0], ylim[1], colors='xkcd:blue', label=r"$\mu$ = 2")


if __name__ == "__main__":
  run_name = sys.argv[1]
  dumpfile = os.path.join("/scratch/03002/bprather/pharm_dumps/M87SimulationLibrary/GRMHD",run_name,"dumps/dump_00001500.h5")
  hdr,geom,dump = io.load_all(dumpfile)
  
  plotfile = os.path.join("/work/03002/bprather/stampede2/movies",run_name,"eht_out.p")
  avg = pickle.load(open(plotfile, "rb"))
  
  # BZ luminosity; see eht_analysis
  if hdr['r_out'] < 100:
    iBZ = i_of(geom, 40) # most SANEs
    rBZ = 40
    rstring = "40"
  else:
    iBZ = i_of(geom, 100) # most MADs
    rBZ = 100
    rstring = "100"
  
  # For converting differentials to theta
  avg['X2'] = geom['X2'][iBZ,:,0]
  Xgeom = np.zeros((4,1,geom['n2']))
  Xgeom[1] = geom['X1'][iBZ,:,0]
  Xgeom[2] = geom['X2'][iBZ,:,0]
  to_dth_bz = dxdX_to_KS(Xgeom, Met.FMKS, geom)[2,2,0]

  ND = avg['t'].shape[0]
  # I can rely on this for now
  start = int(avg['avg_start'])//5
  end = int(avg['avg_end'])//5

  avg['th100'] = geom['th'][iBZ,:,0]
  avg['hth100'] = geom['th'][iBZ,:hdr['n2']//2,0]
  
  # Write OG values for comparison/table import.  To obtain values identical to George's, measure 2 zones outward
  with open("average_"+run_name.replace("/","_").split("x")[0]+".dat", "w") as datf:
    datf.write("# x2 theta dx2/dtheta gdet rho bsq Fem_t Ffl_t F_mass\n")
    for i in range(hdr['n2']):
      datf.write("{} {} {} {} {} {} {} {} {}\n".format(avg['X2'][i], avg['th100'][i], to_dth_bz[i], geom['gdet'][iBZ,i],
                                                       avg['rho_100_th'][i]/hdr['n3'], avg['bsq_100_th'][i]/hdr['n3'],
                                                       -avg['FE_EM_100_th'][i]/hdr['n3'], -avg['FE_Fl_100_th'][i]/hdr['n3'],
                                                       avg['FM_100_th'][i]/hdr['n3']))

  # Add geometric factors to raw sum
  for key in avg.keys():
    if key[-7:] == "_100_th":
      avg[key] *= geom['gdet'][iBZ,:]*hdr['dx3']

  start = int(avg['avg_start'])//5
  end = int(avg['avg_end'])//5
  print("Compare sigma > 1: {} vs {}".format(np.mean(avg['Lj_sigma1'][start:end]),
                                             hdr['dx2']*np.sum(avg['FE_100_th'][np.where(avg['bsq_100_th']/avg['rho_100_th'] > 1)])))
  print("Compare FE > 0: {} vs {}".format(np.mean(avg['Lj_allp'][start:end]),
                                          hdr['dx2']*np.sum(avg['FE_100_th'][np.where(avg['FE_100_th'] > 0)])))
  print("Compare bg > 1: {} vs {}".format(np.mean(avg['Lj_bg1'][start:end]),
                                          hdr['dx2']*np.sum(avg['FE_100_th'][np.where(avg['betagamma_100_th'] > 1)])))

  # Convert for plotting in theta
  for key in avg.keys():
    if key[-7:] == "_100_th":
      avg[key] *= to_dth_bz

  # L_th
  fig, axes = plt.subplots(2,2, figsize=(FIGX, FIGY))

  # Plot Luminosity contribution (incl. KE) as a fn of theta at r=100
  ax = axes[0,0]
  ax.plot(avg['hth100'], avg['FE_100_th'][:hdr['n2']//2], color='C0', label=r"$\frac{d FE_{tot}}{d\theta}$")
  ax.plot(avg['hth100'], avg['FE_EM_100_th'][:hdr['n2']//2], color='C1', label=r"$\frac{d FE_{EM}}{d\theta}$")
  ax.plot(avg['hth100'], avg['FE_Fl_100_th'][:hdr['n2']//2], color='C3', label=r"$\frac{d FE_{Fl}}{d\theta}$")

  prop = np.sum(avg['FE_100_th'][np.where(avg['FE_100_th'] > 0)])/np.max(avg['FE_100_th'])
  Ltot_th_acc = [np.sum(avg['FE_100_th'][:n])/prop for n in range(hdr['n2']//2)]
  LBZ_th_acc = [np.sum(avg['FE_EM_100_th'][:n])/prop for n in range(hdr['n2']//2)]
  ax.plot(avg['hth100'], Ltot_th_acc, 'C4', label=r"Acc. $L_{tot}$")
  ax.plot(avg['hth100'], LBZ_th_acc, 'C5', label=r"Acc. $L_{BZ}$")

  overlay_th_contours(ax,avg)

#  ax.axhline(0.0, color='k', linestyle=':')
  ax.set_ylim([1e-2, None])
  ax.set_yscale('log')
  ax.legend(loc='upper right')
  
  ax = axes[0,1]
  ax.plot(avg['hth100'], avg['FL_100_th'][:hdr['n2']//2], color='C6', label=r"$\frac{d FL_{tot}}{d\theta}$")
  ax.plot(avg['hth100'], avg['FL_EM_100_th'][:hdr['n2']//2], color='C7', label=r"$\frac{d FL_{EM}}{d\theta}$")
  ax.plot(avg['hth100'], avg['FL_Fl_100_th'][:hdr['n2']//2], color='C8', label=r"$\frac{d FL_{Fl}}{d\theta}$")
  ax.set_ylim([1e-2, None])
  ax.set_yscale('log')
  ax.legend(loc='upper right')
  
  ax = axes[1,0]
  ax.plot(avg['hth100'], avg['FM_100_th'][:hdr['n2']//2], color='k', label=r"$\frac{d FM_{tot}}{d\theta}$")
  ax.set_ylim([1e-2, None])
  ax.set_yscale('log')
  ax.legend(loc='upper right')
  
  ax = axes[1,1]
  ax.plot(avg['hth100'], avg['rho_100_th'][:hdr['n2']//2], color='k', label=r"$\rho$")
  ax.set_yscale('log')
  ax.legend(loc='upper right')

  plt.savefig(run_name.replace("/", "_") + '_L_th.png')
  plt.close(fig)
  
  fig, ax = plt.subplots(2,2,figsize=(FIGX, FIGY))
  bplt.plot_thphi(ax[0,0], geom, np.log10(avg['FE_100_thphi']), iBZ, project=False, label = r"FE $\theta-\phi$ slice")
  overlay_thphi_contours(ax[0,0], geom, avg, legend=True)
  bplt.plot_thphi(ax[0,1], geom, np.log10(avg['FM_100_thphi']), iBZ, project=False, label = r"FM $\theta-\phi$ slice")
  overlay_thphi_contours(ax[0,1], geom, avg)
  bplt.plot_thphi(ax[1,0], geom, np.log10(avg['FL_100_thphi']), iBZ, project=False, label = r"FL $\theta-\phi$ slice")
  overlay_thphi_contours(ax[1,0], geom, avg)
  bplt.plot_thphi(ax[1,1], geom, np.log10(avg['rho_100_thphi']), iBZ, project=False, label = r"$\rho$ $\theta-\phi$ slice")
  overlay_thphi_contours(ax[1,1], geom, avg)
  
  plt.savefig(run_name.replace("/", "_") + '_L_100_thphi.png')
  plt.close(fig)
  
  fig, ax = plt.subplots(2,2,figsize=(FIGX, FIGY))
  bplt.plot_xz(ax[0,0], geom, np.log10(avg['FE_rth']), label = "FE X-Z Slice")
  overlay_rth_contours(ax[0,0], geom, avg, legend=True)
  bplt.plot_xz(ax[0,1], geom, np.log10(avg['FM_rth']), label = "FM X-Z Slice")
  overlay_rth_contours(ax[0,1], geom, avg)
  bplt.plot_xz(ax[1,0], geom, np.log10(avg['FL_rth']), label = "FL X-Z Slice")
  overlay_rth_contours(ax[1,0], geom, avg)
  bplt.plot_xz(ax[1,1], geom, np.log10(avg['rho_rth']), label = "RHO X-Z Slice")
  overlay_rth_contours(ax[1,1], geom, avg)
  
  plt.savefig(run_name.replace("/", "_") + '_L_rth.png')
  plt.close(fig)
  
  
  
  
  
  

