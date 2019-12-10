################################################################################
#                                                                              #
#  PLOT ONE PRIMITIVE                                                          #
#                                                                              #
################################################################################

import hdf5_to_dict as io
import plot as bplt
from analysis_fns import *

import matplotlib
import matplotlib.pyplot as plt

import sys
import numpy as np

USEARRSPACE=False
UNITS=False

FIGX = 12
FIGY = 12

# Decide where to measure fluxes
def i_of(rcoord):
  i = 0
  while geom['r'][i,hdr['n2']//2,0] < rcoord:
    i += 1
  i -= 1
  return i

def overlay_thphi_contours(ax, geom, r):
  s = "_" + str(r) + "_thphi"
  r_i = i_of(r)
  max_th = geom['n2']//2
  x = bplt.loop_phi(geom['x'][r_i,:max_th,:])
  y = bplt.loop_phi(geom['y'][r_i,:max_th,:])
  prep = lambda var : bplt.loop_phi(var[r_i,:max_th,:])
  #ax.contour(x,y, prep(dump['ucon']), [0.0], colors='k')
  ax.contour(x,y, prep(dump['sigma']), [1.0], colors='xkcd:blue')
  #ax.contour(x,y, prep(dump['sigma']), [10.0], colors='C3')
  #ax.contour(x,y, prep(dump['Be_b']), [0.02], colors='C4')
  #ax.contour(x,y, prep(dump['Be_b']), [1.0], colors='C5')
  ax.contour(x,y, prep(dump['Be_nob']), [0.02], colors='xkcd:purple')
  ax.contour(x,y, prep(dump['Be_nob']), [1.0], colors='xkcd:green')
  #ax.contour(x,y, prep(geom['r']*dump['ucon'][:,:,:,1]), [1.0], color='C8')
  #ax.contour(x,y, prep(dump['gamma']), [1.5], color='C9')

if len(sys.argv) > 2:
  dumpfile = sys.argv[1]
  gridfile = sys.argv[2]
elif len(sys.argv) > 1:
  dumpfile = sys.argv[1]
  gridfile = None
else:
  print("Specify dump file!")
  exit(-1)

if gridfile is not None:
  hdr = io.load_hdr(dumpfile)
  geom = io.load_geom(hdr, gridfile)
  dump = io.load_dump(dumpfile, hdr, geom)
else:
  # Assumes gridfile in same directory
  hdr,geom,dump = io.load_all(dumpfile)

# BZ luminosity; see eht_analysis
if hdr['r_out'] < 100:
  iBZ = i_of(40) # most SANEs
  rstring="40"
else:
  iBZ = i_of(100) # most MADs
  rstring="100"

# Add bernoulli param to dump to plot/cut
dump['Be_b'] = bernoulli(dump, with_B=True)
dump['Be_nob'] = bernoulli(dump, with_B=False)
dump['sigma'] = dump['bsq']/dump['RHO']

fig, ax = plt.subplots(2,2,figsize=(FIGX, FIGY))

bplt.plot_thphi(ax[0,0], geom, T_mixed(dump, 1, 0)[iBZ,:,:], iBZ, label = "FE 2D Slice r="+rstring)
overlay_thphi_contours(ax[0,0], geom, 100)
bplt.plot_thphi(ax[0,1], geom, dump['RHO'][iBZ,:,:]*dump['ucon'][iBZ,:,:,1], iBZ, label = "FM 2D Slice r="+rstring)
overlay_thphi_contours(ax[0,1], geom, 100)
bplt.plot_thphi(ax[1,0], geom, T_mixed(dump, 1, 3)[iBZ,:,:], iBZ, label = "FL 2D Slice r="+rstring)
overlay_thphi_contours(ax[1,0], geom, 100)
bplt.plot_thphi(ax[1,1], geom, dump['RHO'][iBZ,:,:], iBZ, label = "RHO 2D Slice r="+rstring)
overlay_thphi_contours(ax[1,1], geom, 100)

plt.savefig("_".join(dumpfile.split("/")[-5:-2]) + '_L1_100_thphi.png')
plt.close(fig)

