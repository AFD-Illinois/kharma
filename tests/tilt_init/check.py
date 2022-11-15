#!/usr/bin/env python

# Image the first dump
# Note NO CONSISTENCY CHECKS
import os,sys
import glob
import numpy as np
import matplotlib.pyplot as plt

import pyharm
import pyharm.plots.plot_dumps as hplt

dumpname = "torus.out0.00000.phdf"
dump = pyharm.load_dump(dumpname)
fig, ax = plt.subplots(1,1,figsize=(7,7))
hplt.plot_xz(ax, dump, 'log_beta', window=[-200,200,-200,200])
plt.savefig(dumpname+"_beta.png")

fig, ax = plt.subplots(1,1,figsize=(7,7))
hplt.plot_xz(ax, dump, 'log_bsq', window=[-200,200,-200,200])
plt.savefig(dumpname+"_bsq.png")

fig, ax = plt.subplots(1,1,figsize=(7,7))
hplt.plot_xz(ax, dump, 'log_B1', window=[-200,200,-200,200])
plt.savefig(dumpname+"_B1.png")

fig, ax = plt.subplots(1,1,figsize=(7,7))
hplt.plot_xz(ax, dump, 'log_B2', window=[-200,200,-200,200])
plt.savefig(dumpname+"_B2.png")

fig, ax = plt.subplots(1,1,figsize=(7,7))
hplt.plot_xz(ax, dump, 'log_B3', window=[-200,200,-200,200])
plt.savefig(dumpname+"_B3.png")

fig, ax = plt.subplots(1,1,figsize=(7,7))
hplt.plot_xz(ax, dump, 'log_rho', window=[-200,200,-200,200])
plt.savefig(dumpname+"_rho.png")
