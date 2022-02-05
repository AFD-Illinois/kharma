#!/usr/bin/env python3

################################################################################
#                                                                              #
#  PLOT DIFFERENCES IN TWO FILES                                               #
#                                                                              #
################################################################################

from __future__ import print_function, division

import pyHARM
from pyHARM import parameters
import pyHARM.ana.plot as pplt
import pyHARM.util as util

import os,sys
import numpy as np
import matplotlib.pyplot as plt


USEARRSPACE=True
GHOSTS = False
if USEARRSPACE:
    if GHOSTS:
        window = (-0.1, 1.1, -0.1, 1.1)
    else:
        window = (0, 1, 0, 1)
else:
    SIZE = 40
    window = (-SIZE, SIZE, -SIZE, SIZE)

FIGX = 20
FIGY = 20

dump1file = sys.argv[1]
dump2file = sys.argv[2]
imname = sys.argv[3]

dump1 = pyHARM.load_dump(dump1file, add_ghosts=GHOSTS)
#Hopefully this fails for dumps that shouldn't be compared
dump2 = pyHARM.load_dump(dump2file, add_ghosts=GHOSTS)

N1 = dump1['n1']; N2 = dump1['n2']; N3 = dump1['n3']

log_floor = -12

# TODO properly option log, rel, lim
def plot_diff_xy(ax, var, rel=False, lim=None):
    if rel:
        if lim is not None:
            pplt.plot_xy(ax, dump1, np.abs((dump1[var] - dump2[var])/dump1[var]),
            vmin=0, vmax=lim, label=var, arrayspace=USEARRSPACE, window=window)
        else:
            pplt.plot_xy(ax, dump1, np.abs((dump1[var] - dump2[var])/dump1[var]),
            label=var, arrayspace=USEARRSPACE, window=window)
    else:
        if lim is not None:
            pplt.plot_xy(ax, dump1, np.log10(np.abs(dump1[var] - dump2[var])),
            vmin=log_floor, vmax=lim, label=var, arrayspace=USEARRSPACE, window=window)
        else:
            pplt.plot_xy(ax, dump1, np.log10(np.abs(dump1[var] - dump2[var])),
            vmin=log_floor, vmax=0, label=var, arrayspace=USEARRSPACE, window=window)

def plot_diff_xz(ax, var, rel=False, lim=None):
    if rel:
        if lim is not None:
            pplt.plot_xz(ax, dump1, np.abs((dump1[var] - dump2[var])/dump1[var]),
            vmin=0, vmax=lim, label=var, arrayspace=USEARRSPACE, window=window)
        else:
            pplt.plot_xz(ax, dump1, np.abs((dump1[var] - dump2[var])/dump1[var]),
            label=var, arrayspace=USEARRSPACE, window=window)
    else:
        if lim is not None:
            pplt.plot_xz(ax, dump1, np.log10(np.abs(dump1[var] - dump2[var])),
            vmin=log_floor, vmax=lim, label=var, arrayspace=USEARRSPACE, window=window)
        else:
            pplt.plot_xz(ax, dump1, np.log10(np.abs(dump1[var] - dump2[var])),
            vmin=log_floor, vmax=0, label=var, arrayspace=USEARRSPACE, window=window)

# Plot the difference
nxplot = 3
nyplot = 3
vars = list(dump2['prim_names']) # Parthenon isn't dealing with KEL

fig = plt.figure(figsize=(FIGX, FIGY))
for i,name in enumerate(vars):
  ax = plt.subplot(nyplot, nxplot, i+1)
  plot_diff_xz(ax, name, rel=True) #, lim=1)
  ax.set_xlabel('')
  ax.set_ylabel('')

plt.tight_layout()

plt.savefig(imname+"_xz.png", dpi=100)
plt.close(fig)

if dump1['n3'] > 1:
    fig = plt.figure(figsize=(FIGX, FIGY))
    for i,name in enumerate(vars):
        ax = plt.subplot(nyplot, nxplot, i+1)
        plot_diff_xy(ax, name, rel=True, lim=1)
        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.tight_layout()

    plt.savefig(imname+"_xy.png", dpi=100)
    plt.close(fig)
