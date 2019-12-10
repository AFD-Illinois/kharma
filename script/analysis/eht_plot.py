################################################################################
#                                                                              #
#  PLOTS OF VARIABLES COMPUTED IN eht_analysis.py                              #
#                                                                              #
################################################################################

import matplotlib
import os, sys
import numpy as np
import pickle

import util
import units
from analysis_fns import *

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# For radials
FIGX = 10
FIGY = 10
# For flux plots; per-plot Y dim
PLOTY = 3
SIZE = 40

RADS = True
FLUXES = True
EXTRAS = True
DIAGS = True
OMEGA = False
FLUX_PROF = False
TEMP = False
BSQ = False
MFLUX = False
BFLUX = True
TH_PROFS = True
CFUNCS = True
PSPECS = True
LCS = True
COMPARE = False
PDFS = True
JSQ = True

def i_of(var, coord):
  i = 0
  while var[i] < coord:
    i += 1
  i -= 1
  return i

# Return the portion of a variable which constitutes quiescence
def qui(avg, vname):
  if 'avg_start' in avg:
    istart = i_of(avg['t'], avg['avg_start'])
  else:
    istart = i_of(avg['t'], 5000)
  if 'avg_end' in avg:
    iend = i_of(avg['t'], avg['avg_end'])
  else:
    iend = i_of(avg['t'], 10000)
  return avg[vname][istart:iend]

def print_av_var(vname, tag=None):
  if tag:
    print(tag+":")
  else:
    print(vname+":")
  for label,avg in zip(labels,avgs):
    if vname in avg:
      var_av = np.abs(np.mean(qui(avg,vname)))
      var_std = np.std(qui(avg,vname))
      print("{}: avg {:.3}, std abs {:.3} rel {:.3}".format(label, var_av, var_std, var_std/var_av))

def plot_multi(ax, iname, varname, varname_pretty, logx=False, logy=False, xlim=None, ylim=None, timelabels=False, label_list=None, linestyle='-'):

  if label_list is None: label_list = labels

  for i, avg in enumerate(avgs):
    if varname in avg:
      if avg[iname].size > avg[varname].size:
        # Some vars are only to half in theta
        ax.plot(avg[iname][:avg[iname].size//2], avg[varname], styles[i]+linestyle, label=label_list[i])
      else:
        ax.plot(avg[iname][np.nonzero(avg[varname])], avg[varname][np.nonzero(avg[varname])], styles[i]+linestyle, label=label_list[i])
        if iname == 't':
          startx = (avg['avg_start'] - ti) / (tf - ti)
          endx = (avg['avg_end'] - ti) / (tf - ti)
          ax.axhline(np.mean(qui(avg, varname)), startx, endx, color=styles[i], linestyle='--')
  # Plot additions
  if logx: ax.set_xscale('log')
  if logy: ax.set_yscale('log')
  if ylim is not None: ax.set_ylim(ylim)
  if xlim is not None: ax.set_xlim(xlim)
  ax.grid(True)
  ax.set_ylabel(varname_pretty)
  # Defaults and labels for different plot types:
  if iname == 't':
    if xlim is None:
      ax.set_xlim([ti,tf])
    if timelabels:
      ax.set_xlabel('t/M')
    else:
      ax.set_xticklabels([])
  elif 'freq' in iname:
    if timelabels:
      ax.set_xlabel('Frequency (1/M)')
    else:
      ax.set_xticklabels([])
  elif 'lambda' in iname:
    if timelabels:
      ax.set_xlabel('Correlation time (M)')
    else:
      ax.set_xticklabels([])
  elif iname == 'th':
    ax.set_xlabel(r"$\theta$")
  elif iname == 'r':
    ax.set_xlabel("r")
    if xlim is None:
      ax.set_xlim([0,50]) # For EHT comparison
  if logy:
    ylim = ax.get_ylim()
    if ylim[0] < 1e-5*ylim[1]:
      ax.set_ylim([1e-5*ylim[1], ylim[1]])

def plot_temp():
  fig, ax = plt.subplots(1,1, figsize=(FIGX, FIGY))
  if avgs[0]['r'][-1] > 50:
    txlim = [1e0, 1e3]
  else:
    txlim = [1e0, 1e2]
  
  fit_labs = []
  for i,avg in enumerate(avgs):
    cgs = units.get_cgs()

    # We can't very well plot a temp we don't know
    if 'Pg_r' not in avg:
      return

    avg['Tp_r'] = cgs['MP'] * avg['Pg_r'] / (cgs['KBOL'] * avg['rho_r']) * cgs['CL']**2

    # Add the fits. Aaaaaalll the fits
    x = avg['r'][i_of(avg['r'], 3):i_of(avg['r'], 30)]
    y = avg['Tp_r'][i_of(avg['r'], 3):i_of(avg['r'], 30)]

    coeffs = np.polyfit(np.log(x), np.log(y), deg=1)
    poly = np.poly1d(coeffs)
    yfit = lambda xf: np.exp(poly(np.log(xf)))

    avg['r_fit'] = x
    avg['Tp_r_fit'] = yfit(x)
    fit_lab = r"{:.2g} * r^{:.2g}".format(np.exp(coeffs[1]), coeffs[0])
    print(labels[i], " Ti fit: ", fit_lab)
    fit_labs.append(fit_lab)

  # Plot the profiles themselves
  plot_multi(ax, 'r', 'Tp_r', r"$<T_{i}>$", logx=True, xlim=txlim, logy=True)
  plot_multi(ax, 'r_fit', 'Tp_r_fit', r"$<T_{i}>$", logx=True, xlim=txlim, logy=True, label_list=fit_labs, linestyle='--')

  if len(labels) > 1:
    ax.legend(loc='lower right')
  else:
    ax.set_title(labels[0])
  plt.savefig(fname_out + "_Ti.png")
  plt.close(fig)

def plot_bsq_rise():
  nplot = 1
  fig, ax = plt.subplots(nplot,1, figsize=(FIGX, nplot*PLOTY))
  for avg in avgs:
    if 'B_rt' in avg:
      avg['MagE'] = np.mean(avg['B_rt']**2, axis=-1)

    plot_multi(ax, 't', 'MagE', r"$<B^2>$", logy=True, xlim=[0,10000])

  if len(labels) > 1:
    ax.legend(loc='lower right')
  else:
    ax.set_title(labels[0])
  plt.savefig(fname_out + '_bsq_rise.png')
  plt.close(fig)

def plot_ravgs():
  fig, ax = plt.subplots(3, 3, figsize=(FIGX, FIGY))
  for avg in avgs:
    if 'beta_r' in avg:
      avg['betainv_r'] = 1/avg['beta_r']
    if 'Pg_r' in avg:
      avg['Tp_r'] = avg['Pg_r'] / avg['rho_r']
    if 'B_r' in avg:
      avg['sigma_r'] = avg['B_r']**2 / avg['rho_r']

  plot_multi(ax[0, 0], 'r', 'rho_r', r"$<\rho>$", logy=True) #, ylim=[1.e-2, 1.e0])
  plot_multi(ax[0, 1], 'r', 'Pg_r', r"$<P_g>$", logy=True) #, ylim=[1.e-6, 1.e-2])
  plot_multi(ax[0, 2], 'r', 'Ptot_r', r"$<P_{tot}>$", logy=True) #, ylim=[1.e-6, 1.e-2])
  plot_multi(ax[1, 0], 'r', 'B_r', r"$<|B|>$", logy=True) #, ylim=[1.e-4, 1.e-1])
  plot_multi(ax[1, 1], 'r', 'u^phi_r', r"$<u^{\phi}>$", logy=True) #, ylim=[1.e-3, 1.e1])
  plot_multi(ax[1, 2], 'r', 'u_phi_r', r"$<u_{\phi}>$", logy=True) #, ylim=[1.e-3, 1.e1])
  plot_multi(ax[2, 0], 'r', 'Tp_r', r"$<T>$", logy=True) #, ylim=[1.e-6, 1.e-2])
  plot_multi(ax[2, 1], 'r', 'betainv_r', r"$<\beta^{-1}>$", logy=True) #, ylim=[1.e-2, 1.e1])
  plot_multi(ax[2, 2], 'r', 'sigma_r', r"$<\sigma>$", logy=True) #, ylim=[1.e-2, 1.e1])

  if len(labels) > 1:
    ax[0, -1].legend(loc='upper right')
  else:
    fig.suptitle(labels[0])

  #pad = 0.05
  #plt.subplots_adjust(left=pad, right=1-pad/2, bottom=pad, top=1-pad)
  plt.subplots_adjust(wspace=0.35)
  plt.savefig(fname_out + '_ravgs.png')
  plt.close(fig)

def plot_mflux():
  fig, ax = plt.subplots(2,2, figsize=(0.66*FIGX, 0.66*FIGY))
  
  for avg in avgs:
    if 'outflow_r' in avg:
      avg['outflow_r'] /= avg['Mdot_av']
    if 'FM_jet_r' in avg:
      avg['fm_jet_r'] = avg['FM_jet_r']/avg['Mdot_av']
    if 'FM_r' in avg:
      avg['fm_r'] = avg['FM_r']/avg['Mdot_av']

  plot_multi(ax[0,0], 'r', 'outflow_r', r"$\frac{FM_{out}}{\langle \dot{M} \rangle}$", ylim=[0,3], xlim=[1,30], logx=True)

  plot_multi(ax[1,0], 'r', 'fm_jet_r', r"$\frac{FM_{jet}}{\langle \dot{M} \rangle}$", xlim=[1,1000], logx=True)
  plot_multi(ax[1,1], 'r', 'fm_r', r"$\frac{FM_{tot}}{\langle \dot{M} \rangle}$", xlim=[1,1000], logx=True)
  
  if len(labels) > 1:
    ax[0,0].legend(loc='upper right')
  else:
    fig.suptitle(labels[0])

  plt.savefig(fname_out + '_Mfluxr.png')
  plt.close(fig)

def plot_Bflux():
  fig, ax = plt.subplots(2,2, figsize=(FIGX, FIGY))
  
  for avg in avgs:
    if ('Phi_sph_r' in avg) and ('Phi_mid_r' in avg):
      avg['phi_sph_r'] = avg['Phi_sph_r'] / avg['Mdot_av']
      avg['phi_mid_r'] = avg['Phi_mid_r'] / avg['Mdot_av']
      avg['phi_diff_r'] = avg['phi_sph_r'] - avg['phi_mid_r']
      if 'rho_r' in avgs:
        avg['rho_enc_r'] = np.zeros_like(avg['rho_r'])
        for i in range(avg['rho_r'].size):
          avg['rho_enc_r'][i] = np.sum(avg['rho_r'][:i])
        avg['phi_brnorm_r'] = avg['Phi_mid_r'] / avg['rho_enc_r']

  plot_multi(ax[0, 0], 'r', 'phi_sph_r', r"$\frac{\Phi_{sph}}{\langle \dot{M} \rangle}$")
  plot_multi(ax[0, 1], 'r', 'phi_mid_r', r"$\frac{\Phi_{mid}}{\langle \dot{M} \rangle}$")
  plot_multi(ax[1, 0], 'r', 'phi_diff_r', r"$\frac{\Phi_{sph} - \Phi_{mid}}{\langle \dot{M} \rangle}$")
  plot_multi(ax[1, 0], 'r', 'phi_brnorm_r', r"$\frac{\Phi_{mid}}{\rho_{enc}}$")

  if len(labels) > 1:
    ax[0,0].legend(loc='upper right')
  else:
    fig.suptitle(labels[0])

  pad = 0.05
  plt.subplots_adjust(left=pad, right=1-pad/2, bottom=pad, top=1-pad)
  plt.savefig(fname_out + '_Bfluxr.png')
  plt.close(fig)

def plot_fluxes():
  nplot = 7
  fig,ax = plt.subplots(nplot, 1, figsize=(FIGX, nplot*PLOTY))

  plot_multi(ax[0], 't', 'Mdot', r"$\dot{M}$")
  plot_multi(ax[1], 't', 'Phi_b', r"$\Phi$")
  plot_multi(ax[2], 't', 'Ldot', r"$\dot{L}$")

  for avg in avgs:
    if 'Edot' in avg.keys() and 'Mdot' in avg.keys():
      avg['MmE'] = avg['Mdot'] - avg['Edot']
  plot_multi(ax[3], 't', 'MmE', r"$\dot{M} - \dot{E}$")

  for avg in avgs:
    if 'LBZ_bg1' in avg.keys():
      avg['aLBZ'] = np.abs(avg['LBZ_bg1'])
  plot_multi(ax[4], 't', 'aLBZ', "BZ Luminosity")

  for avg in avgs:
    if 'Lj_bg1' in avg.keys():
      avg['aLj'] = np.abs(avg['Lj_bg1'])
  plot_multi(ax[5], 't', 'aLj', "Jet Luminosity", timelabels=True)

  #plot_multi(ax[6], 't', 'Lum', "Luminosity proxy", timelabels=True)

  if len(labels) > 1:
    ax[0].legend(loc='upper left')
  else:
    ax[0].set_title(labels[0])

  plt.savefig(fname_out + '_fluxes.png')
  plt.close(fig)

  for avg in avgs:
    if 'Mdot' not in avg.keys():
      avg['Mdot_av'] = 1
    else:
      avg['Mdot_av'] = np.mean(qui(avg,'Mdot'))

  nplot = 3
  fig, ax = plt.subplots(nplot,1, figsize=(FIGX, nplot*PLOTY))
#  plot_multi(ax[0], 't', 'Mdot', r"$\dot{M}$")
#  print_av_var('Mdot')

  for avg in avgs:
    if 'Phi_b' in avg.keys():
      avg['phi_b'] = avg['Phi_b']/np.sqrt(avg['Mdot_av'])
  plot_multi(ax[0], 't', 'phi_b', r"$\frac{\Phi_{BH}}{\sqrt{\langle \dot{M} \rangle}}$")
  print_av_var('phi_b', "Normalized Phi_BH")

  for avg in avgs:
    if 'Ldot' in avg.keys():
      avg['ldot'] = np.fabs(avg['Ldot'])/avg['Mdot_av']
  plot_multi(ax[1], 't', 'ldot', r"$\frac{Ldot}{\langle \dot{M} \rangle}$")
  print_av_var('ldot', "Normalized Ldot")

#  for avg in avgs:
#    if 'Edot' in avg.keys():
#      avg['mmE'] = (avg['Mdot'] - avg['Edot'])/avg['Mdot_av']
#  plot_multi(ax[3], 't', 'mmE', r"$\frac{\dot{M} - \dot{E}}{\langle \dot{M} \rangle}$")
#  print_av_var('mmE', "Normalized Mdot-Edot")
  
  for avg in avgs:
    if 'Edot' in avg.keys():
      avg['edot'] = avg['Edot']/avg['Mdot_av']
  plot_multi(ax[2], 't', 'edot', r"$\frac{\dot{E}}{\langle \dot{M} \rangle}$", timelabels=True)
  print_av_var('edot', "Normalized Edot")
  
  for avg in avgs:
    if 'aLBZ' in avg.keys() and 'Mdot' in avg.keys():
      avg['alBZ'] = avg['aLBZ']/avg['Mdot_av']
#  plot_multi(ax[5], 't', 'alBZ', r"$\frac{L_{BZ}}{\langle \dot{M} \rangle}$")
#  print_av_var('alBZ', "Normalized BZ Jet Power")
  
  for avg in avgs:
    if 'aLj' in avg.keys() and 'Mdot' in avg.keys():
      avg['alj'] = avg['aLj']/avg['Mdot_av']
#  plot_multi(ax[6], 't', 'alj', r"$\frac{L_{jet}}{\langle \dot{M} \rangle}$")
#  print_av_var('alj', "Normalized Total Jet Power")

#  for avg in avgs:
#    if 'Lum' in avg.keys():
#      avg['lum'] = np.fabs(avg['Lum'])/avg['Mdot_av']
#  plot_multi(ax[7], 't', 'lum', r"$\frac{Lum}{|\dot{M}|}$", timelabels=True)
#  print_av_var('lum', "Normalized Luminosity Proxy")
  
  if len(labels) > 1:
    ax[0].legend(loc='upper left')
  else:
    ax[0].set_title(labels[0])

  plt.savefig(fname_out + '_normfluxes.png')
  plt.close(fig)

def plot_pspecs():
  spec_keys = ['mdot', 'phi', 'edot', 'ldot', 'alBZ', 'alj', 'lightcurve']
  pretty_keys = [r"$\dot{M}$",
                 r"$\frac{\Phi_{BH}}{\sqrt{\langle \dot{M} \rangle}}$",
                 r"$\frac{\dot{E}}{\langle \dot{M} \rangle}$",
                 r"$\frac{\dot{L}}{\langle \dot{M} \rangle}$",
                 r"$\frac{L_{BZ}}{\langle \dot{M} \rangle}$",
                 r"$\frac{L_{jet}}{\langle \dot{M} \rangle}$",
                 r"ipole lightcurve"
                 ]
  nplot = len(spec_keys)
  fig, ax = plt.subplots((nplot+1)//2, 2, figsize=(1.5*FIGX, nplot*PLOTY))

  for avg in avgs:
    for key in spec_keys:
      # Use the diag version if available for higher time res
      if 'diags' in avg and avg['diags'] is not None and key in avg['diags']:
        avg[key+'_pspec'], avg[key+'_ps_freq'] = pspec(avg['diags'][key], avg['diags']['t'])
      elif key in avg:
        avg[key+'_pspec'], avg[key + '_ps_freq'] = pspec(avg[key], avg['t'])
      # If these happened add a normalized version
      if key+'_pspec' in avg:
        avg[key+'_pspec_f2'] = avg[key+'_pspec'] * avg[key + '_ps_freq']**2

  for i,key in enumerate(spec_keys):
    if key+'_pspec' in avgs[-1]:
      # psmax = np.max(avgs[-1][key+'_pspec'])
      # fmax = np.max(avgs[-1][key + '_ps_freq'])
      plot_multi(ax[i//2, i%2], key+'_ps_freq', key+'_pspec_f2', pretty_keys[i],
                 logx=True, #xlim=[1e-5 * fmax, fmax],
                 logy=True, #ylim=[1e-8 * psmax, psmax],
                 timelabels=True)

  if len(labels) > 1:
    ax[0, 0].legend(loc='upper right')
  else:
    fig.suptitle(labels[0])

  plt.savefig(fname_out + '_pspecs.png')
  plt.close(fig)

def plot_lcs():
  nplot = 1
  fig,ax = plt.subplots(nplot,1, figsize=(FIGX, nplot*PLOTY))

  for avg,fname in zip(avgs,fnames):
    fpaths = [os.path.join(os.path.dirname(fname), "163", "m_1_1_20", "lightcurve.dat"),
    os.path.join(os.path.dirname(fname), "17", "m_1_1_20", "lightcurve.dat")]
    for fpath in fpaths:
      print(fpath)
      if os.path.exists(fpath):
        print("Found ",fpath)
        cols = np.loadtxt(fpath).transpose()
        # Normalize to 2000 elements
        f_len = cols.shape[1]
        t_len = avg['t'].size
        if f_len >= t_len:
          avg['lightcurve'] = cols[2][:t_len]
          avg['lightcurve_pol'] = cols[1][:t_len]
        elif f_len < t_len:
          avg['lightcurve'] = np.zeros(t_len)
          avg['lightcurve_pol'] = np.zeros(t_len)
          avg['lightcurve'][:f_len] = cols[2]
          avg['lightcurve'][f_len:] = avg['lightcurve'][f_len-1]
          avg['lightcurve_pol'][:f_len] = cols[1]
          avg['lightcurve_pol'][f_len:] = avg['lightcurve_pol'][f_len-1]

  plot_multi(ax, 't', 'lightcurve', r"ipole lightcurve", timelabels=True)
  print_av_var('lightcurve', "Lightcurve from ipole")

  if len(labels) > 1:
    ax.legend(loc='upper left')
  else:
    fig.suptitle(labels[0])

  plt.savefig(fname_out + '_lcs.png')
  plt.close(fig)

def plot_extras():
  nplot = 2
  fig, ax = plt.subplots(nplot,1, figsize=(FIGX, nplot*PLOTY))

  # Efficiency explicitly as a percentage
  for i,avg in enumerate(avgs):
    if 'Edot' in avg.keys():
      avg['Eff'] = (avg['Mdot'] + avg['Edot'])/avg['Mdot_av']*100
  plot_multi(ax[0], 't', 'Eff', "Efficiency (%)", ylim=[-10,200])

  plot_multi(ax[1], 't', 'Edot', r"$\dot{E}$", timelabels=True)

  if len(labels) > 1:
    ax[0].legend(loc='upper left')
  else:
    ax[0].set_title(labels[0])

  plt.savefig(fname_out + '_extras.png')
  plt.close(fig)

def plot_diags():
  nplot = 7
  fig, ax = plt.subplots(nplot,1, figsize=(FIGX, nplot*PLOTY))
  ax = ax.flatten()

  plot_multi(ax[0], 't', 'Etot', "Total E")
  plot_multi(ax[1], 't', 'sigma_max', r"$\sigma_{max}$")
  plot_multi(ax[2], 't', 'betainv_max', r"$\beta^{-1}_{max}$")
  plot_multi(ax[3], 't', 'Theta_max', r"$\Theta_{max}$")

  plot_multi(ax[4], 't', 'rho_min', r"$\rho_{min}$")
  plot_multi(ax[5], 't', 'U_min', r"$U_{min}$")

  # TODO include HARM's own diagnostics somehow? Re-insert just this one?
  plot_multi(ax[6], 't_d', 'divbmax_d', "max divB", timelabels=True)

  if len(labels) > 1:
    ax[1].legend(loc='lower right')
  else:
    ax[1].set_title(labels[0])

  plt.savefig(fname_out + '_diagnostics.png')
  plt.close(fig)

def plot_omega():
  # Omega
  fig, ax = plt.subplots(2,1, figsize=(FIGX, FIGY))
  # Renormalize omega to omega/Omega_H for plotting
  for avg in avgs:
    if 'omega_hth' in avg.keys(): #Then both are
      avg['omega_hth'] *= 4/avg['a']
      avg['omega_av_hth'] *= 4/avg['a']
  plot_multi(ax[0], 'th_5', 'omega_hth', r"$\omega_f$/$\Omega_H$ (EH, single shell)", ylim=[-1,2])
  plot_multi(ax[1], 'th_5', 'omega_av_hth', r"$\omega_f$/$\Omega_H$ (EH, 5-zone average)", ylim=[-1,2])

  # Legend
  if len(labels) > 1:
    ax[0].legend(loc='lower left')
  else:
    ax[0].set_title(labels[0])

  # Horizontal guidelines
  for a in ax.flatten():
    a.axhline(0.5, linestyle='--', color='k')

  plt.savefig(fname_out + '_omega.png')
  plt.close(fig)

def plot_th_profs():
  # Resolution-dependence of values in midplane
  fig, ax = plt.subplots(1,2, figsize=(FIGX, FIGY/3))
  plot_multi(ax[0], 'th_eh', 'betainv_25_th', r"$\beta^{-1} (r = 25)$", logy=True)
  plot_multi(ax[1], 'th_eh', 'sigma_25_th', r"$\sigma (r = 25)$", logy=True)
  # Legend
  if len(labels) > 1:
    ax[0].legend(loc='lower left')
  else:
    ax[0].set_title(labels[0])

  plt.savefig(fname_out + '_th_profs.png')
  plt.close(fig)

def plot_cfs():
  # Correlation functions in midplane
  fig, ax = plt.subplots(2, 2, figsize=(FIGX, FIGY))
  for avg in avgs:
    if np.max(avg['rho_cf_rphi']) > 2:
      del avg['rho_cf_rphi'], avg['betainv_cf_rphi'], avg['rho_cf_10_phi'], avg['betainv_cf_10_phi']
    if 'rho_cf_rphi' in avg:
      avg['rho_cl_r'] = corr_length(avg['rho_cf_rphi'])
      avg['rho_cf_10_phi'] = avg['rho_cf_rphi'][i_of(avg['r'], 10)]
    if 'betainv_cf_rphi' in avg:
      avg['betainv_cl_r'] = corr_length(avg['betainv_cf_rphi'])
      avg['betainv_cf_10_phi'] = avg['betainv_cf_rphi'][i_of(avg['r'], 10)]

  plot_multi(ax[0, 0], 'phi', 'rho_cf_10_phi', r"$\bar{R}(\rho) (r = 10)$", xlim=[0, np.pi])
  plot_multi(ax[0, 1], 'phi', 'betainv_cf_10_phi', r"$\bar{R}(\beta^{-1}) (r = 10)$", xlim=[0, np.pi])
  plot_multi(ax[1, 0], 'r', 'rho_cl_r', r"$\lambda (\rho, r)$", logx=True, xlim=[1, 500])
  plot_multi(ax[1, 1], 'r', 'betainv_cl_r', r"$\lambda (\rho, r)$", logx=True, xlim=[1, 500])
  # Legend
  if len(labels) > 1:
    ax[0,1].legend(loc='lower left')
  else:
    fig.suptitle(labels[0])

  plt.savefig(fname_out + '_cls.png')
  plt.close(fig)

def plot_flux_profs():
  # For converting to theta
  Xgeom = np.zeros((4,1,geom['n2']))
  Xgeom[1] = avg['r'][iBZ]
  Xgeom[2] = avg['th_100']
  to_th = 1/dxdX_to_KS(Xgeom, Met.FMKS, geom)[2,2,1]

  for avg in avgs:
    if 'FE_100_th' in avg.keys(): # Then all are
      avg['FE_100_th'] *= to_th
      avg['FE_Fl_100_th'] *= to_th
      avg['FE_EM_100_th'] *= to_th

  plot_multi(ax[0,0], 'th_100', 'FE_100_th', r"$\frac{dFE}{d\theta}$ ($r = 100$)")
  plot_multi(ax[0,1], 'th_100', 'FE_Fl_100_th', r"$\frac{dFE_{Fl}}{d\theta}$ ($r = 100$)")
  plot_multi(ax[1,0], 'th_100', 'FE_EM_100_th', r"$\frac{dFE_{EM}}{d\theta}$ ($r = 100$)")

  # Legend
  ax[0,0].legend(loc='lower left')

  plt.savefig(fname_out + '_flux_profs.png')
  plt.close(fig)

def plot_var_compare():
  nplotsy, nplotsx = 2,2
  fig, ax = plt.subplots(nplotsy, nplotsx, figsize=(FIGX, FIGY))

  for i,vname in enumerate(['phi_b', 'ldot', 'edot', 'lightcurve']):
    stddevs = [np.std(qui(avg,vname))/np.abs(np.mean(qui(avg,vname))) for avg in avgs if vname in avg.keys()]
    n2s = [int(fname.split("x")[1]) for fname in fnames]
    axis = ax[i//nplotsy, i % nplotsx]
    axis.plot(n2s, stddevs, marker='o', color='k')
    axis.set_xscale('log')
    axis.set_xlabel(r"$N_{\theta}$")
    axis.set_ylim([0,None])
    axis.set_title("Relative variance of "+vname)


  plt.savefig(fname_out + '_var_compare.png')
  plt.close(fig)

def plot_pdfs():
  nplotsy, nplotsx = 2, 1
  fig, ax = plt.subplots(nplotsy, nplotsx, figsize=(FIGX, FIGY))
  pdf_vars = []
  for avg in avgs:
    avg['pdf_bins'] = np.linspace(-3.5, 3.5, 200)
    for var in avg:
      if var[-4:] == '_pdf' and var not in pdf_vars:
        pdf_vars.append(var)

  for i,var in enumerate(pdf_vars):
    plot_multi(ax[i], 'pdf_bins', var, var)

  # Legend
  if len(labels) > 1:
    ax[0].legend(loc='upper right')
  else:
    fig.suptitle(labels[0])

  plt.savefig(fname_out + '_pdfs.png')
  plt.close(fig)

def plot_jsq():
  nplotsy, nplotsx = 1, 1
  fig, ax = plt.subplots(nplotsy, nplotsx, figsize=(FIGX, PLOTY))
  for avg in avgs:
    if 'Jsqtot_rt' in avg:
      avg['Jsqtot_t'] = np.sum(avg['Jsqtot_rt'], axis=-1)

  plot_multi(ax, 't', 'Jsqtot_t', r"Total $J^2$ on grid", timelabels=True)
  print_av_var('Jsqtot_t', "Total J^2")
  

  plt.savefig(fname_out + '_jsq.png')
  plt.close(fig)

if __name__ == "__main__":
  if len(sys.argv) < 3:
    util.warn('Format: python eht_plot.py analysis_output [analysis_output ...] [labels_list] image_name')
    sys.exit()

  # All interior arguments are files to overplot, except possibly the last
  if len(sys.argv) < 4:
    last_file = -1
  else:
    last_file = -2


  fnames = sys.argv[1:last_file]
  avgs = []
  for filename in fnames:
    # Encoding arg is for python2 numpy bytestrings
    avgs.append(pickle.load(open(filename,'rb'), encoding = 'latin1'))

  # Split the labels, or use the output name as a label
  if len(sys.argv) > 3:
    labels = sys.argv[-2].split(",")
  else:
    labels = [sys.argv[-1].replace("_"," ")]

  if len(labels) < len(avgs):
    util.warn("Too few labels!")
    sys.exit()

  fname_out = sys.argv[-1]

  # For time plots.  Also take MAD/SANE for axes?
  #ti = avgs[0]['t'][0]
  nt = avgs[0]['t'].size
  ti = avgs[0]['t'][nt//2]
  tf = avgs[0]['t'][-1]

  # Default styles
  if len(avgs) > 1:
    styles = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
  else:
    styles = ['k']
    
  if RADS: plot_ravgs()
  if FLUXES: plot_fluxes()
  if EXTRAS: plot_extras()
  if DIAGS: plot_diags()
  if OMEGA: plot_omega()
  if BSQ: plot_bsq_rise()
  if TEMP: plot_temp()
  if MFLUX: plot_mflux()
  if BFLUX: plot_Bflux()
  if TH_PROFS: plot_th_profs()
  if LCS: plot_lcs()
  if CFUNCS: plot_cfs()
  if PSPECS: plot_pspecs()
  if PDFS: plot_pdfs()
  if JSQ: plot_jsq()
  if len(avgs) == 1:
    if FLUX_PROF: plot_flux_profs()
  else:
    if COMPARE: plot_var_compare()


