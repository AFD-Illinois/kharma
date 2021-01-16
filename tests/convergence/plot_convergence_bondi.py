#!/usr/bin/env python3

# Bondi problem convergence plots

import os,sys
import numpy as np
import matplotlib.pyplot as plt

import pyHARM
from pyHARM.parameters import parse_parthenon_dat

RES = [int(x) for x in sys.argv[1].split(",")]
BASE = "../../"
LONG = sys.argv[2]
SHORT = sys.argv[3]

L1 = []

# 3d is kind of pointless even in KHARMA...
# for res in [32, 64]

# 2d
for res in RES:
    params = parse_parthenon_dat(BASE+"pars/bondi.par")
    params['n1'] = params['n1tot'] = params['nx1'] = res
    params['n2'] = params['n2tot'] = params['nx2'] = res
    start = pyHARM.load_dump("bondi_2d_{}_start_{}.phdf".format(res, SHORT), params=params)
    end = pyHARM.load_dump("bondi_2d_{}_end_{}.phdf".format(res, SHORT), params=params)

    r = start['r'][:,start['n2']//2]

    imin = 0
    while r[imin] < params['r_eh']:
        imin += 1

    rho0 = np.mean(start['RHO'][imin:,:,0], axis=1)
    rho1 = np.mean(end['RHO'][imin:,:,0], axis=1)

    L1.append(np.mean(np.fabs(rho1 - rho0)))

# MEASURE CONVERGENCE
L1 = np.array(L1)
powerfit = np.polyfit(np.log(RES), np.log(L1), 1)[0]
print("Powerfit: {} L1: {}".format(powerfit, L1))

# MAKE PLOTS
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.plot(RES, L1, marker='s', label='RHO')

amp = L1[0]*RES[0]*RES[0]
xmin = RES[0]/2.
xmax = RES[-1]*2.
ax.plot([xmin, xmax], amp*np.asarray([xmin, xmax])**-2., color='k', linestyle='--', label='N^-2')

plt.xscale('log', base=2); plt.yscale('log')
plt.xlim([RES[0]/np.sqrt(2.), RES[-1]*np.sqrt(2.)])
plt.xlabel('N'); plt.ylabel('L1')
plt.title("Bondi test convergence, {}".format(LONG))
plt.legend(loc=1)
plt.savefig("convergence_bondi_{}.png".format(SHORT))

