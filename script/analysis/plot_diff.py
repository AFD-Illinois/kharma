################################################################################
#                                                                              #
#  PLOT DIFFERENCES IN TWO FILES                                               #
#                                                                              #
################################################################################

from __future__ import print_function, division

import plot as bplt
import util
import hdf5_to_dict as io

import os,sys
import numpy as np
import matplotlib.pyplot as plt


USEARRSPACE=True
NLINES = 20
SIZE = 600

FIGX = 20
FIGY = 16

dump1file = sys.argv[1]
dump2file = sys.argv[2]
imname = sys.argv[3]

hdr, geom, dump1 = io.load_all(dump1file, derived_vars=False)
#Hopefully this fails for dumps that shouldn't be compared
dump2 = io.load_dump(dump2file, hdr, geom, derived_vars=False)

N1 = hdr['n1']; N2 = hdr['n2']; N3 = hdr['n3']

log_floor = -60

# TODO properly option log, rel, lim
def plot_diff_xy(ax, var, rel=False, lim=None):
    if rel:
        if lim is not None:
            bplt.plot_xy(ax, geom, np.abs((dump1[var] - dump2[var])/dump1[var]), vmin=0, vmax=lim, label=var, cbar=False, arrayspace=USEARRSPACE)
        else:
            bplt.plot_xy(ax, geom, np.abs((dump1[var] - dump2[var])/dump1[var]), label=var, cbar=False, arrayspace=USEARRSPACE)
    else:
        if lim is not None:
            bplt.plot_xy(ax, geom, np.log10(np.abs(dump1[var] - dump2[var])), vmin=log_floor, vmax=lim, label=var, cbar=False, arrayspace=USEARRSPACE)
        else:
            bplt.plot_xy(ax, geom, np.log10(np.abs(dump1[var] - dump2[var])), vmin=log_floor, vmax=0, label=var, cbar=False, arrayspace=USEARRSPACE)

def plot_diff_xz(ax, var, rel=False, lim=None):
    if rel:
        if lim is not None:
            bplt.plot_xz(ax, geom, np.abs((dump1[var] - dump2[var])/dump1[var]), vmin=0, vmax=lim, label=var, cbar=False, arrayspace=USEARRSPACE)
        else:
            bplt.plot_xz(ax, geom, np.abs((dump1[var] - dump2[var])/dump1[var]), label=var, cbar=False, arrayspace=USEARRSPACE)
    else:
        if lim is not None:
            bplt.plot_xz(ax, geom, np.log10(np.abs(dump1[var] - dump2[var])), vmin=log_floor, vmax=lim, label=var, cbar=False, arrayspace=USEARRSPACE)
        else:
            bplt.plot_xz(ax, geom, np.log10(np.abs(dump1[var] - dump2[var])), vmin=log_floor, vmax=0, label=var, cbar=False, arrayspace=USEARRSPACE)

# Plot the difference
nxplot = 4
nyplot = 3
vars = list(hdr['prim_names'])+['fail','divB']

fig = plt.figure(figsize=(FIGX, FIGY))
for i,name in enumerate(vars):
  ax = plt.subplot(nyplot, nxplot, i+1)
  plot_diff_xy(ax, name)
  ax.set_xlabel('')
  ax.set_ylabel('')

plt.tight_layout()

plt.savefig(imname+"_xy.png", dpi=100)
plt.close(fig)

fig = plt.figure(figsize=(FIGX, FIGY))
for i,name in enumerate(vars):
  ax = plt.subplot(nyplot, nxplot, i+1)
  plot_diff_xz(ax, name)
  ax.set_xlabel('')
  ax.set_ylabel('')

plt.tight_layout()

plt.savefig(imname+"_xz.png", dpi=100)
plt.close(fig)
