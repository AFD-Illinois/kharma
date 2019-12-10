################################################################################
#                                                                              # 
#  CALCULATE TIME-DEPENDENT AND TIME-AVERAGED QUANTITIES                       #
#                                                                              # 
################################################################################

from __future__ import print_function, division

from analysis_fns import *
import hdf5_to_dict as io
import util

import os, sys
import multiprocessing
import psutil
import pickle
import itertools

import numpy as np

# Option to calculate fluxes at (just inside) r = 5
# This reduces interference from floors
floor_workaround_flux = False
# Option to ignore accretion at high magnetization (funnel)
# This also reduces interference from floors
floor_workaround_funnel = False

# Whether to calculate each expensive set of variables
# Once performed once, calculations will be ported to each new output file
calc_ravgs = True
calc_basic = True
calc_jet_profile = False
calc_jet_cuts = True
calc_lumproxy = True
calc_etot = True
calc_efluxes = False
calc_outfluxes = False

calc_pdfs = True
pdf_nbins = 200

if len(sys.argv) < 2:
  util.warn('Format: python eht_analysis.py /path/to/dumps [start time] [start radial averages] [stop radial averages] [stop time]')
  sys.exit()

# This doesn't seem like the _right_ way to do optional args
# Skips everything before tstart, averages between tavg_start and tavg_end
tstart = None
tavg_start = None
tavg_end = None
tend = None
if sys.argv[1] == "-d":
  debug = True
  path = sys.argv[2]
  if len(sys.argv) > 3:
    tstart = float(sys.argv[3])
  if len(sys.argv) > 4:
    tavg_start = float(sys.argv[4])
  if len(sys.argv) > 5:
    tavg_end = float(sys.argv[5])
  if len(sys.argv) > 6:
    tend = float(sys.argv[6])
else:
  debug = False
  path = sys.argv[1]
  if len(sys.argv) > 2:
    tstart = float(sys.argv[2])
  if len(sys.argv) > 3:
    tavg_start = float(sys.argv[3])
  if len(sys.argv) > 4:
    tavg_end = float(sys.argv[4])
  if len(sys.argv) > 5:
    tend = float(sys.argv[5])

dumps = io.get_dumps_list(path)
ND = len(dumps)

hdr = io.load_hdr(dumps[0])
geom = io.load_geom(hdr, path)

if tstart is None:
  tstart = 0.

# If the time after which to average wasn't given, just use the back half of dumps
if tavg_start is None:
  tavg_start = io.get_dump_time(dumps[ND//2]) - 0.1
# Sometimes we don't know times (i.e. above will be 0) but want averages
# We always want to average over all dumps in these cases
if tavg_start < 0.:
  tavg_start = 0.

if tavg_end is None:
  tavg_end = io.get_dump_time(dumps[-1])
if tavg_end == 0.:
  tavg_end = float(ND)

if tend is None:
  tend = io.get_dump_time(dumps[-1])
if tend == 0.:
  tend = float(ND)

# Leave several extra zones if using MKS3 coordinates
if geom['metric'] == "MKS3":
  iEH = i_of(geom, hdr['r_eh'])+4
else:
  iEH = i_of(geom, hdr['r_eh'])

if floor_workaround_flux:
  iF = i_of(geom, 5) # Measure fluxes at r=5M
else:
  iF = iEH

# Max radius when computing "total" energy
iEmax = i_of(geom, 40)

# BZ luminosity
# 100M seems like the standard measuring spot (or at least, BHAC does it that way)
# L_BZ seems constant* after that, but much higher within ~50M
if geom['r_out'] < 100 or geom['r'][-1,geom['n2']//2,0] < 100: # If in theory or practice the sim is small...
  iBZ = i_of(geom, 40) # most SANEs
else:
  iBZ = i_of(geom, 100) # most MADs

jmin, jmax = get_j_vals(geom)

print("Running from t={} to {}, averaging from {} to {}".format(tstart, tend, tavg_start, tavg_end))
print("Using EH at zone {}, Fluxes at zone {}, Emax within zone {}, L_BZ at zone {}".format(iEH, iF, iEmax, iBZ))

def avg_dump(n):
  out = {}

  out['t'] = io.get_dump_time(dumps[n])
  # When we don't know times, fudge
  if out['t'] == 0 and n != 0:
    out['t'] = n

  if out['t'] < tstart or out['t'] > tend:
    #print("Loaded {} / {}: {} (SKIPPED)".format((n+1), len(dumps), out['t']))
    # Still return the time
    return out
  else:
    print("Loaded {} / {}: {}".format((n+1), len(dumps), out['t']))
    dump = io.load_dump(dumps[n], hdr, geom, extras=False)

  # EHT Radial profiles: special fn for profile, averaged over phi, 1/3 theta, time
  if calc_ravgs:
    for var in ['rho', 'Theta', 'B', 'Pg', 'Ptot', 'beta', 'u^phi', 'u_phi', 'sigma', 'FM']:
      out[var+'_rt'] = eht_profile(geom, d_fns[var](dump), jmin, jmax)
      out[var+'_jet_rt'] = eht_profile(geom, d_fns[var](dump), 0, jmin) + eht_profile(geom, d_fns[var](dump), jmax, geom['n2'])
      if out['t'] >= tavg_start and out['t'] <= tavg_end:
        out[var+'_r'] = out[var+'_rt']
        out[var+'_jet_r'] = out[var+'_jet_rt']
  
    if out['t'] >= tavg_start and out['t'] <= tavg_end:
      # CORRELATION FUNCTION
      for var in ['rho', 'betainv']:
        Rvar = corr_midplane(geom, d_fns[var](dump))
        out[var+'_cf_rphi'] = Rvar
        out[var+'_cf_10_phi'] = Rvar[i_of(geom,10),:]
      
      # THETA AVERAGES
      for var in ['betainv', 'sigma']:
        out[var+'_25_th'] = theta_av(geom, d_fns[var](dump), i_of(geom, 25), 5, fold=False)
      
      # These are divided averages, not average of division, so not amenable to d_fns
      Fcov01, Fcov13 = Fcov(geom, dump, 0, 1), Fcov(geom, dump, 1, 3)
      out['omega_hth'] = theta_av(geom, Fcov01, iEH, 1) / theta_av(geom, Fcov13, iEH, 1)
      out['omega_av_hth'] = theta_av(geom, Fcov01, iEH, 5) / theta_av(geom, Fcov13, iEH, 5)
  
      # This produces much worse results
      #out['omega_alt_hth'] = theta_av(Fcov(dump, 0, 2), iEH, 1) / theta_av(Fcov(dump, 2, 3), iEH, 1)
      #out['omega_alt_av_hth'] = theta_av(Fcov(dump, 0, 2), iEH-2, 5) / theta_av(Fcov(dump, 2, 3), iEH-2, 5)

  if calc_basic:
    # FIELD STRENGTHS
    # The HARM B_unit is sqrt(4pi)*c*sqrt(rho) which has caused issues:
    #norm = np.sqrt(4*np.pi) # This is what I believe matches T,N,M '11 and Narayan '12
    norm = 1 # This is what the EHT comparison uses?
  
    if geom['mixed_metrics']:
      # When different, B1 will be in the _vector_ coordinates.  Must perform the integral in those instead of zone coords
      # Some gymnastics were done to keep in-memory size small
      dxEH = np.einsum("i,...ij->...j", np.array([0, geom['dx1'], geom['dx2'], geom['dx3']]), np.linalg.inv(geom['vec_to_grid'][iEH,:,:,:]))
      out['Phi_b'] = 0.5*norm * np.sum( np.fabs(dump['B1'][iEH,:,:]) * geom['gdet_vec'][iEH,:,None]*dxEH[:,None,2]*dxEH[:,None,3], axis=(0,1) )
    else:
      out['Phi_sph_r'] = 0.5*norm*sum_shell(geom, np.fabs(dump['B1']))
      out['Phi_b'] = out['Phi_sph_r'][iEH]
      
      out['Phi_mid_r'] = np.zeros_like(out['Phi_sph_r'])
      for i in range(geom['n1']):
        out['Phi_mid_r'][i] = norm*sum_plane(geom, -dump['B2'], within=i)
  
    # FLUXES
    # Radial profiles of Mdot and Edot, and their particular values
    # EHT code-comparison normalization has all these values positive
    for var,flux in [['Edot','FE'],['Mdot','FM'],['Ldot','FL']]:
      if out['t'] >= tavg_start and out['t'] <= tavg_end:
        out[flux+'_r'] = sum_shell(geom, d_fns[flux](dump))
      out[var] = sum_shell(geom, d_fns[flux](dump), at_zone=iF)
    # Mdot and Edot are defined inward
    out['Mdot'] *= -1
    out['Edot'] *= -1

    # Maxima (for gauging floors)
    for var in ['sigma', 'betainv', 'Theta']:
      out[var+'_max'] = np.max(d_fns[var](dump))
    # Minima
    for var in ['rho', 'U']:
      out[var+'_min'] = np.min(d_fns[var](dump))
    # TODO KEL? plot in "floor space"?  Full set of energy ratios?

  # Profiles of different fluxes to gauge jet power calculations
  if calc_jet_profile:
    for var in ['rho', 'bsq', 'FM', 'FE', 'FE_EM', 'FE_Fl', 'FL', 'FL_EM', 'FL_Fl', 'betagamma', 'Be_nob', 'Be_b', 'mu']:
      out[var+'_100_tht'] = np.sum(d_fns[var](dump)[iBZ], axis=-1)
      if out['t'] >= tavg_start and out['t'] <= tavg_end:
        out[var+'_100_th'] = out[var+'_100_tht']
        out[var+'_100_thphi'] = d_fns[var](dump)[iBZ,:,:]
        out[var+'_rth'] = d_fns[var](dump).mean(axis=-1)

  # Blandford-Znajek Luminosity L_BZ
  # This is a lot of luminosities!
  if calc_jet_cuts:
    # TODO cut on phi/t averages? -- needs 2-pass cut...
    cuts = {'sigma1' : lambda dump : (d_fns['sigma'](dump) > 1),
            #'sigma10' : lambda dump : (d_fns['sigma'](dump) > 10),
            'Be_b0' : lambda dump : (d_fns['Be_b'](dump) > 0.02),
            'Be_b1' : lambda dump : (d_fns['Be_b'](dump) > 1),
            'Be_nob0' : lambda dump : (d_fns['Be_nob'](dump) > 0.02),
            'Be_nob1' : lambda dump : (d_fns['Be_nob'](dump) > 1),
            #'mu1' : lambda dump : (d_fns['mu'](dump) > 1),
            #'mu2' : lambda dump : (d_fns['mu'](dump) > 2),
            #'mu3' : lambda dump : (d_fns['mu'](dump) > 3),
            'bg1' : lambda dump : (d_fns['betagamma'](dump) > 1.0),
            'bg05' : lambda dump : (d_fns['betagamma'](dump) > 0.5),
            'allp' : lambda dump : (d_fns['FE'](dump) > 0)}
  
    # Terminology:
    # LBZ = E&M energy only, any cut
    # Lj = full E flux, any cut
    # Ltot = Lj_allp = full luminosity wherever it is positive
    for lum,flux in [['LBZ', 'FE_EM'], ['Lj', 'FE']]:
      for cut in cuts.keys():
        out[lum+'_'+cut+'_rt'] = sum_shell(geom, d_fns[flux](dump), mask=cuts[cut](dump))
        if out['t'] >= tavg_start and out['t'] <= tavg_end:
          out[lum+'_'+cut+'_r'] = out[lum+'_'+cut+'_rt']
        out[lum+'_'+cut] = out[lum+'_'+cut+'_rt'][iBZ]

  if calc_lumproxy:
    rho, Pg, B = d_fns['rho'](dump), d_fns['Pg'](dump), d_fns['B'](dump)
    # See EHT code comparison paper
    j = rho**3 / Pg**2 * np.exp(-0.2 * (rho**2 / ( B * Pg**2))**(1./3.))
    out['Lum_rt'] = eht_profile(geom, j, jmin, jmax)

  if calc_etot:
    # Total energy and current, summed by shells to allow cuts on radius
    for tot_name, var_name in [['Etot', 'JE0']]:
      out[tot_name+'_rt'] = sum_shell(geom, d_fns[var_name](dump))
    for tot_name, var_name in [['Jsqtot', 'jsq']]:
      out[tot_name+'_rt'] = sum_shell(geom, d_fns[var_name](geom, dump))

  if calc_efluxes:
    # Conserved (maybe; in steady state) 2D energy flux
    for var in ['JE0', 'JE1', 'JE2']:
      out[var+'_rt'] = sum_shell(geom, d_fns[var](dump))
      if out['t'] >= tavg_start and out['t'] <= tavg_end:
        out[var+'_rth'] = d_fns[var](dump).mean(axis=-1)

  # Total outflowing portions of variables
  if calc_outfluxes:
    for name,var in [['outflow', 'FM'], ['outEflow', 'FE']]:
      var_temp = d_fns[var](dump)
      out[name+'_rt'] = sum_shell(geom, var_temp, mask=(var_temp > 0))
      if out['t'] >= tavg_start and out['t'] <= tavg_end:
        out[name+'_r'] = out[name+'_rt']
  
  if calc_pdfs:
    for var in ['betainv', 'rho']:
      out[var+'_pdf'], _ = np.histogram(np.log10(d_fns[var](dump)),
                                            bins=pdf_nbins, range=(-3.5, 3.5),
                                            weights=np.repeat(geom['gdet'], geom['n3']).reshape((geom['n1'], geom['n2'], geom['n3'])),
                                            density=True)

  dump.clear()
  del dump
  
  return out

def merge_dict(n, out, out_full):
  # Merge the output dicts
  # TODO write to an HDF5 file incrementally?
  for key in list(out.keys()):
    if key not in out_full:
      if key[-3:] == '_rt':
        out_full[key] = np.zeros((ND, hdr['n1']))
      elif key[-5:] == '_htht':
        out_full[key] = np.zeros((ND, hdr['n2']//2))
      elif key[-4:] == '_tht':
        out_full[key] = np.zeros((ND, hdr['n2']))
      elif key[-5:] == '_rtht':
        out_full[key] = np.zeros((ND, hdr['n1'], hdr['n2']))
      elif key[-7:] == '_thphit':
        out_full[key] = np.zeros((ND, hdr['n2'], hdr['n3']))
      elif key[-5:] == '_pdft':
        out_full[key] = np.zeros((ND, pdf_nbins))
      elif (key[-2:] == '_r' or key[-4:] == '_hth' or key[-3:] == '_th' or key[-4:] == '_phi' or
            key[-4:] == '_rth' or key[-5:] == '_rphi' or key[-6:] == '_thphi' or key[-4:] == '_pdf'):
        out_full[key] = np.zeros_like(out[key])
      else:
        out_full[key] = np.zeros(ND)
    if (key[-2:] == '_r' or key[-4:] == '_hth' or key[-3:] == '_th' or key[-4:] == '_phi' or
        key[-4:] == '_rth' or key[-5:] == '_rphi' or key[-6:] == '_thphi' or key[-4:] == '_pdf'):
      # Weight the average correctly for _us_.  Full weighting will be done on merge w/'avg_w'
      if my_avg_range > 0:
        out_full[key] += out[key]/my_avg_range
    else:
      out_full[key][n] = out[key]

# TODO this, properly, some other day
if ND < 200:
  nstart, nmin, nmax, nend = 0, 0, ND-1, ND-1
elif ND < 300:
  nstart, nmin, nmax, nend = 0, ND//2, ND-1, ND-1
else:
  nstart, nmin, nmax, nend = int(tstart)//5, int(tavg_start)//5, int(tavg_end)//5, int(tend)//5

full_avg_range = nmax - nmin

if nmin < nstart: nmin = nstart
if nmin > nend: nmin = nend
if nmax < nstart: nmax = nstart
if nmax > nend: nmax = nend

my_avg_range = nmax - nmin

print("nstart = {}, nmin = {}, nmax = {} nend = {}".format(nstart,nmin,nmax,nend))

# Make a dict for merged variables, throw in what we know now to make merging easier
out_full = {}
for key in ['a', 'gam', 'gam_e', 'gam_p']:
  out_full[key] = hdr[key]

# Toss in the common geom lists and our weight in the overall average
out_full['r'] = geom['r'][:,hdr['n2']//2,0]

# For quick angular plots. Note most will need geometry to convert from dX2 to dth
out_full['th_eh'] = geom['th'][iEH,:,0]
out_full['th_bz'] = geom['th'][iBZ,:,0]

out_full['phi'] = geom['phi'][0,hdr['n2']//2,:]

out_full['avg_start'] = tavg_start
out_full['avg_end'] = tavg_end
out_full['avg_w'] = my_avg_range / full_avg_range
print("Will weight averages by {}".format(out_full['avg_w']))

# Fill the output dict with all per-dump or averaged stuff
# Hopefully in a way that doesn't keep too much of it around in memory
nthreads = util.calc_nthreads(hdr, pad=0.2)
util.iter_parallel(avg_dump, merge_dict, out_full, ND, nthreads)

# Add divBmax from HARM's own diagnostic output, if available.  We can recompute the rest, but not this
diag = io.load_log(path)
if diag is not None:
  out_full['t_d'] = diag['t']
  out_full['divbmax_d'] = diag['divbmax']

# Deduce the name of the output file
if tstart > 0 or tend < 10000:
  outfname = "eht_out_{}_{}.p".format(tstart,tend)
else:
  outfname = "eht_out.p"
# See if there's anything already there we're not calculating, and import it
if os.path.exists(outfname):
  with open(outfname, "rb") as prev_file:
    out_old = pickle.load(prev_file)
    for key in out_old:
      if key not in out_full:
        out_full[key] = out_old[key]

# OUTPUT
with open(outfname, "wb") as outf:
  print("Writing {}".format(outfname))
  pickle.dump(out_full, outf)
