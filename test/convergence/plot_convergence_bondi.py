#!/usr/bin/env python3

# Plot "convergence" i.e. stability vs resolution of a 2D Bondi problem

import sys
sys.path.append("../../scripts/")
import phdf

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

RES = [int(el) for el in sys.argv[1:]]
L1 = np.zeros(len(RES))

for m in range(len(RES)):
  os.chdir('../dumps_' + str(RES[m]))

  f0 = phdf(np.sort(glob.glob("dump_*.h5"))[0])
  f1 = phdf(np.sort(glob.glob("dump_*.h5"))[-1])

  imin = 0
  while np.exp(f0.xf[:,imin]) < 2.5:
    imin += 1

  rho0 = np.mean(f0.Get('c.c.bulk.prims')[:,0,:,imin:,0], axis=-1)
  rho1 = np.mean(f1.Get('c.c.bulk.prims')[:,0,:,imin:,0], axis=-1)

  L1[m] = np.mean(np.fabs(rho1 - rho0))

# MEASURE CONVERGENCE
powerfit = np.polyfit(np.log(RES), np.log(L1), 1)[0]
print("Powerfit: {} L1: {}".format(powerfit, L1))

os.chdir('../plots/')

# MAKE PLOTS
fig = plt.figure(figsize=(16.18,10))

ax = fig.add_subplot(1,1,1)
ax.plot(RES, L1, marker='s', label='RHO')

amp = 1.0e-3
ax.plot([RES[0]/2., RES[-1]*2.],
  10.*amp*np.asarray([RES[0]/2., RES[-1]*2.])**-2.,
  color='k', linestyle='--', label='N^-2')
plt.xscale('log', basex=2); plt.yscale('log')
plt.xlim([RES[0]/np.sqrt(2.), RES[-1]*np.sqrt(2.)])
plt.xlabel('N'); plt.ylabel('L1')
plt.title("Bondi flow convergence")
plt.legend(loc=1)
plt.savefig('bondi.png', bbox_inches='tight')

