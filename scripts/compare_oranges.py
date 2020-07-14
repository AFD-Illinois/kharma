################################################################################
#                                                                              #
#  PLOT DIFFERENCES IN TWO FILES                                               #
#                                                                              #
################################################################################

from __future__ import print_function, division

import pyHARM
from pyHARM import parameters
import pyHARM.ana.plot as bplt
import pyHARM.ana.util as util

import os,sys
import numpy as np
import matplotlib.pyplot as plt


USEARRSPACE=True
if USEARRSPACE:
    SIZE = 1
    window = [0, SIZE, 0, SIZE]
else:
    SIZE = 40
    window = [-SIZE, SIZE, -SIZE, SIZE]

FIGX = 20
FIGY = 20

dump1file = sys.argv[1]
parfile = sys.argv[2]
dump2file = sys.argv[3]
imname = sys.argv[4]

dump1 = pyHARM.load_dump(dump1file)
#Hopefully this fails for dumps that shouldn't be compared
params = {}
parameters.parse_parthenon_dat(params, parfile)
parameters.fix(params)
dump2 = pyHARM.load_dump(dump2file, params=params)

N1 = dump1['n1']; N2 = dump1['n2']; N3 = dump1['n3']

log_floor = -2

# TODO properly option log, rel, lim
def plot_diff_xy(ax, var, rel=False, lim=None):
    if rel:
        if lim is not None:
            bplt.plot_xy(ax, dump1, np.abs((dump1[var] - dump2[var])/dump1[var]),
            vmin=0, vmax=lim, label=var, cbar=False, arrayspace=USEARRSPACE, window=window)
        else:
            bplt.plot_xy(ax, dump1, np.abs((dump1[var] - dump2[var])/dump1[var]),
            label=var, cbar=False, arrayspace=USEARRSPACE, window=window)
    else:
        if lim is not None:
            bplt.plot_xy(ax, dump1, np.log10(np.abs(dump1[var] - dump2[var])),
            vmin=log_floor, vmax=lim, label=var, cbar=False, arrayspace=USEARRSPACE, window=window)
        else:
            bplt.plot_xy(ax, dump1, np.log10(np.abs(dump1[var] - dump2[var])),
            vmin=log_floor, vmax=0, label=var, cbar=False, arrayspace=USEARRSPACE, window=window)

def plot_diff_xz(ax, var, rel=False, lim=None):
    if rel:
        if lim is not None:
            bplt.plot_xz(ax, dump1, np.abs((dump1[var] - dump2[var])/dump1[var]),
            vmin=0, vmax=lim, label=var, cbar=False, arrayspace=USEARRSPACE, window=window)
        else:
            bplt.plot_xz(ax, dump1, np.abs((dump1[var] - dump2[var])/dump1[var]),
            label=var, cbar=False, arrayspace=USEARRSPACE, window=window)
    else:
        if lim is not None:
            bplt.plot_xz(ax, dump1, np.log10(np.abs(dump1[var] - dump2[var])),
            vmin=log_floor, vmax=lim, label=var, cbar=False, arrayspace=USEARRSPACE, window=window)
        else:
            bplt.plot_xz(ax, dump1, np.log10(np.abs(dump1[var] - dump2[var])),
            vmin=log_floor, vmax=0, label=var, cbar=False, arrayspace=USEARRSPACE, window=window)

# Plot the difference
nxplot = 3
nyplot = 3
vars = list(dump2['prim_names']) # Parthenon isn't dealing with KEL

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
