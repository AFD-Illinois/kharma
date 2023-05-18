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
    try:
        fig, ax = plt.subplots(1,1,figsize=(7,7))
        hplt.plot_xz(ax, dump, 'U1', log=True, arrayspace=True, window=[0,1,0,1])
        fig.savefig(dumpname+"_U1.png")
        plt.close(fig)
    except:
        print("Error plotting U1 of {}".format(dumpname))

    fig, ax = plt.subplots(1,1,figsize=(4,7))
    hplt.plot_xz(ax, dump, 'rho', log=True, window=(-10,0,-10,10))
    hplt.overlay_field(ax, dump, nlines=8)
    fig.savefig(dumpname+"_rho.png")
    plt.close(fig)
    del dump
