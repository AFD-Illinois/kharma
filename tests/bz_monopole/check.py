#!/usr/bin/env python

import os,sys
import glob
import numpy as np
import matplotlib.pyplot as plt

import pyharm
import pyharm.plots.plot_dumps as hplt

# Plots ONLY; no automated failures
for dumpname in np.sort(glob.glob("bz_monopole.out0.*.phdf")):
    dump = pyharm.load_dump(dumpname)
    fig, ax = plt.subplots(1,1,figsize=(7,7))
    hplt.plot_xz(ax, dump, 'log_U1', arrayspace=True, window=[0,1,0,1])
    plt.savefig(dumpname+"_U1.png")

    fig, ax = plt.subplots(1,1,figsize=(4,7))
    hplt.plot_xz(ax, dump, 'rho', window=(-10,0,-10,10))
    hplt.overlay_field(ax, dump, nlines=8)
    plt.savefig(dumpname+"_rho.png")
