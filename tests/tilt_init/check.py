#!/usr/bin/env python3

# Image the first dump
import os,sys
import glob
import numpy as np
import matplotlib.pyplot as plt

import pyHARM
import pyHARM.ana.plot as hplt

dumpname = "torus.out0.00000.phdf"
dump = pyHARM.load_dump(dumpname, calc_derived=True)
fig, ax = plt.subplots(1,1,figsize=(7,7))
hplt.plot_xz(ax, dump, 'log_beta', window=[-200,200,-200,200])
plt.savefig(dumpname+"_beta.png")

fig, ax = plt.subplots(1,1,figsize=(7,7))
hplt.plot_xz(ax, dump, 'log_rho', window=[-200,200,-200,200])
plt.savefig(dumpname+"_rho.png")