################################################################################
#                                                                              #
#  PLOT ONE PRIMITIVE                                                          #
#                                                                              #
################################################################################

import hdf5_to_dict as io
import plot as bplt
from analysis_fns import *
import units

import matplotlib
import matplotlib.pyplot as plt

import sys
import numpy as np
from scipy.signal import convolve2d

# TODO parse lots of options I set here
USEARRSPACE=False
UNITS=False

SIZE = 100
window=[-SIZE,SIZE,-SIZE,SIZE]
#window=[-SIZE/4,SIZE/4,0,SIZE]
FIGX = 10
FIGY = 10

dumpfile = sys.argv[1]
if len(sys.argv) > 3:
  gridfile = sys.argv[2]
  var = sys.argv[3]
elif len(sys.argv) > 2:
  gridfile = None
  var = sys.argv[2]

# Optionally take extra name
name = sys.argv[-1]

if UNITS and var not in ['Tp']:
  M_unit = float(sys.argv[-1])

if gridfile is not None:
  hdr = io.load_hdr(dumpfile)
  geom = io.load_geom(hdr, gridfile)
  dump = io.load_dump(dumpfile, hdr, geom)
else:
  # Assumes gridfile in same directory
  hdr,geom,dump = io.load_all(dumpfile)

# If we're plotting a derived variable, calculate + add it
if var in ['jcov', 'jsq']:
  dump['jcov'] = np.einsum("...i,...ij->...j", dump['jcon'], geom['gcov'][:,:,None,:,n])
  dump['jsq'] = np.sum(dump['jcon']*dump['jcov'], axis=-1)
elif var in ['divE2D']:
  JE1g, JE2g = T_mixed(dump, 1,0).mean(axis=-1)*geom['gdet'], T_mixed(dump, 2,0).mean(axis=-1)*geom['gdet']
  face_JE1 = 0.5*(JE1g[:-1,:] + JE1g[1:,:])
  face_JE2 = 0.5*(JE2g[:,:-1] + JE2g[:,1:])
  divJE = (face_JE1[1:,1:-1] - face_JE1[:-1,1:-1]) / geom['dx1'] + (face_JE2[1:-1,1:] - face_JE2[1:-1,:-1]) / geom['dx2']
  dump[var] = np.zeros_like(dump['RHO'])
  dump[var][1:-1,1:-1,0] = divJE
  dump[var] /= np.sqrt(T_mixed(dump, 1,0)**2 + T_mixed(dump, 2,0)**2 + T_mixed(dump, 3,0)**2)*geom['gdet'][:,:,None]
elif var in ['divB2D']:
  B1g, B2g = dump['B1'].mean(axis=-1)*geom['gdet'], dump['B2'].mean(axis=-1)*geom['gdet']
  corner_B1 = 0.5*(B1g[:,1:] + B1g[:,:-1])
  corner_B2 = 0.5*(B2g[1:,:] + B2g[:-1,:])
  divB = (corner_B1[1:,:] - corner_B1[:-1,:]) / geom['dx1'] + (corner_B2[:,1:] - corner_B2[:,:-1]) / geom['dx2']
  dump[var] = np.zeros_like(dump['RHO'])
  dump[var][:-1,:-1,0] = divB
  dump[var] /= np.sqrt(dump['B1']**2 + dump['B2']**2 + dump['B3']**2)*geom['gdet'][:,:,None]
elif var in ['divB3D']:
  B1g, B2g, B3g = dump['B1']*geom['gdet'][:,:,None], dump['B2']*geom['gdet'][:,:,None], dump['B3']*geom['gdet'][:,:,None]
  corner_B1 = 0.25*(B1g[:,1:,1:] + B1g[:,1:,:-1] + B1g[:,:-1,1:] + B1g[:,:-1,:-1])
  corner_B2 = 0.25*(B2g[1:,:,1:] + B2g[1:,:,:-1] + B2g[:-1,:,1:] + B2g[:-1,:,:-1])
  corner_B3 = 0.25*(B3g[1:,1:,:] + B3g[1:,:-1,:] + B3g[:-1,1:,:] + B3g[:-1,:-1,:])
  divB = (corner_B1[1:,:,:] - corner_B1[:-1,:,:]) / geom['dx1'] + (corner_B2[:,1:,:] - corner_B2[:,:-1,:]) / geom['dx2'] + (corner_B3[:,:,1:] - corner_B3[:,:,:-1]) / geom['dx3']
  dump[var] = np.zeros_like(dump['RHO'])
  dump[var][:-1,:-1,:-1] = divB
  dump[var] /= np.sqrt(dump['B1']**2 + dump['B2']**2 + dump['B3']**2)*geom['gdet'][:,:,None]
elif var[-4:] == "_pdf":
  var_og = var[:-4]
  dump[var_og] = d_fns[var_og](dump)
  dump[var], dump[var+'_bins'] = np.histogram(np.log10(dump[var_og]), bins=200, range=(-3.5,3.5), weights=np.repeat(geom['gdet'], geom['n3']).reshape(dump[var_og].shape), density=True)
elif var not in dump:
  dump[var] = d_fns[var](dump)


# Add units after all calculations, manually
if UNITS:
  if var in ['Tp']:
    cgs = units.get_cgs()
    dump[var] /= cgs['KBOL']
  else:
    unit = units.get_units_M87(M_unit, tp_over_te=3)

    if var in ['bsq']:
      dump[var] *= unit['B_unit']**2
    elif var in ['B']:
      dump[var] *= unit['B_unit']
    elif var in ['Ne']:
      dump[var] = dump['RHO'] * unit['Ne_unit']
    elif var in ['Te']:
      dump[var] = ref['ME'] * ref['CL']**2 * unit['Thetae_unit'] * dump['UU']/dump['RHO']
    elif var in ['Thetae']:
      # TODO non-const te
      dump[var] = unit['Thetae_unit'] * dump['UU']/dump['RHO']

fig = plt.figure(figsize=(FIGX, FIGY))
# Treat PDFs separately
if var[-4:] == "_pdf":
  plt.plot(dump[var+'_bins'][:-1], dump[var])
  plt.title("PDF of "+var[:-4])
  plt.xlabel("Log10 value")
  plt.ylabel("Probability")
  
  plt.savefig(name+".png", dpi=100)
  plt.close(fig)
  exit()

# Plot XY differently for vectors, scalars
if var in ['jcon','ucon','ucov','bcon','bcov']:
  axes = [plt.subplot(2, 2, i) for i in range(1,5)]
  for n in range(4):
    bplt.plot_xy(axes[n], geom, np.log10(dump[var][:,:,:,n]), arrayspace=USEARRSPACE, window=window)
elif var not in ['divE2D', 'divB2D']:
  # TODO allow specifying vmin/max, average from command line or above
  ax = plt.subplot(1, 1, 1)
  bplt.plot_xy(ax, geom, dump[var], arrayspace=USEARRSPACE, window=window, vmin=1e10, vmax=1e12)

plt.tight_layout()

plt.savefig(name+"_xy.png", dpi=100)
plt.close(fig)

fig = plt.figure(figsize=(FIGX, FIGY))

# Plot XZ
if var in ['jcon', 'ucon', 'ucov', 'bcon', 'bcov']:
  axes = [plt.subplot(2, 2, i) for i in range(1, 5)]
  for n in range(4):
    bplt.plot_xz(axes[n], geom, np.log10(dump[var][:,:,:,n]), arrayspace=USEARRSPACE, window=window)

elif var in ['divB2D', 'divE2D', 'divE2D_face', 'divB3D']:
  ax = plt.subplot(1, 1, 1)
  bplt.plot_xz(ax, geom, np.log10(np.abs(dump[var])), arrayspace=USEARRSPACE, window=window, vmin=-6, vmax=0)
  if var in ['divE2D', 'divE2D_face']:
    #JE1 = -T_mixed(dump, 1,0)
    #JE2 = -T_mixed(dump, 2,0)
    JE1 = dump['ucon'][:,:,:,1]
    JE2 = dump['ucon'][:,:,:,2]
    bplt.overlay_flowlines(ax, geom, JE1, JE2, nlines=20, arrayspace=USEARRSPACE)
    #bplt.overlay_quiver(ax, geom, JE1, JE2)
  else:
    bplt.overlay_field(ax, geom, dump, nlines=20, arrayspace=USEARRSPACE)
else:
  ax = plt.subplot(1, 1, 1)
  bplt.plot_xz(ax, geom, np.log10(dump[var]), vmin=-3, vmax=1, arrayspace=USEARRSPACE, window=window)
  norm = np.sqrt(dump['ucon'][:,:,0,1]**2 + dump['ucon'][:,:,0,2]**2)*geom['gdet']
  JF1 = dump['ucon'][:,:,:,1] #/norm
  JF2 = dump['ucon'][:,:,:,2] #/norm
  
  #bplt.overlay_quiver(ax, geom, dump, JF1, JF2, cadence=96, norm=15)
  bplt.overlay_flowlines(ax, geom, JF1, JF2, nlines=100, arrayspace=USEARRSPACE, reverse=True)

plt.tight_layout()

plt.savefig(name+"_xz.png", dpi=100)
plt.close(fig)
