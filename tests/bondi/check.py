#!/usr/bin/env python

# Bondi problem convergence plots

import os,sys
import numpy as np
import matplotlib.pyplot as plt

import pyharm

RES = [int(x) for x in sys.argv[1].split(",")]
LONG = sys.argv[2]
SHORT = sys.argv[3]

L1 = []

# 2d
for res in RES:
    start = pyharm.load_dump("bondi_2d_{}_start_{}.phdf".format(res, SHORT))
    end = pyharm.load_dump("bondi_2d_{}_end_{}.phdf".format(res, SHORT))
    params = start.params

    r = start['r'][:,start['n2']//2]

    imin = 0
    while r[imin] < params['r_eh']:
        imin += 1

    r = r[imin:]

    rho0 = np.mean(start['RHO'][imin:,:], axis=1)
    rho1 = np.mean(end['RHO'][imin:,:], axis=1)

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(r, rho0, label='Initial')
    ax.plot(r, rho1, label='Final')
    plt.xlabel('r'); plt.ylabel('rho')
    plt.title("Bondi test stability, {}".format(LONG))
    plt.legend()
    plt.savefig("bondi_compare_{}_{}.png".format(res, SHORT))

    L1.append(np.mean(np.fabs(rho1 - rho0)))

# MEASURE CONVERGENCE
L1 = np.array(L1)
powerfit = np.polyfit(np.log(RES), np.log(L1), 1)[0]
print("Powerfit: {} L1: {}".format(powerfit, L1))

fail = 0
if powerfit < -2.2 or powerfit > -1.9:
    fail = 1

# MAKE PLOTS
fig = plt.figure(figsize=(5,5))
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

exit(fail)
