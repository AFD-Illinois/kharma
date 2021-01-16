#!/usr/bin/env python3

# MHD linear modes convergence plots
import os,sys
import numpy as np
import matplotlib.pyplot as plt

import pyHARM
from pyHARM.parameters import parse_parthenon_dat

RES = [int(x) for x in sys.argv[1].split(",")]
BASE = "../../"
LONG = sys.argv[2]
SHORT = sys.argv[3]

NVAR = 8
VARS = ['rho', 'u', 'u1', 'u2', 'u3', 'B1', 'B2', 'B3']

amp = 1.e-4
k1 = 2.*np.pi
k2 = 2.*np.pi
k3 = 2.*np.pi
var0 = np.zeros(NVAR)

# Background
var0[0] = 1.
var0[1] = 1.
# Magnetic field
var0[5] = 1.
var0[6] = 0.
var0[7] = 0.

L1 = []

# EIGENMODES
dvar = np.zeros(NVAR)
if "entropy" in SHORT:
    dvar[0] = 1.
if "slow" in SHORT:
    dvar[0] = 0.556500332363
    dvar[1] = 0.742000443151
    dvar[2] = -0.282334999306
    dvar[3] = 0.0367010491491
    dvar[4] = 0.0367010491491
    dvar[5] = -0.195509141461
    dvar[6] = 0.0977545707307
    dvar[7] = 0.0977545707307
if "alfven" in SHORT:
    dvar[3] =  -0.339683110243
    dvar[4] =  0.339683110243
    dvar[6] =  0.620173672946
    dvar[7] =  -0.620173672946
if "fast" in SHORT:
    dvar[0]  =  0.481846076323
    dvar[1]    =  0.642461435098
    dvar[2]   =  -0.0832240462505
    dvar[3]   =  -0.224080007379
    dvar[4]   =  -0.224080007379
    dvar[5]   =  0.406380545676
    dvar[6]   =  -0.203190272838
    dvar[7]   =  -0.203190272838
dvar *= amp

# USE DUMPS IN FOLDERS OF GIVEN FORMAT
for m, res in enumerate(RES):
    params = parse_parthenon_dat(BASE+"pars/mhdmodes.par")
    params['n1'] = params['n1tot'] = params['nx1'] = res
    params['n2'] = params['n2tot'] = params['nx2'] = res
    params['n3'] = params['n3tot'] = params['nx3'] = res
    dump = pyHARM.load_dump("mhd_3d_{}_end_{}.phdf".format(res, SHORT), params=params)

    X1 = dump['x']
    X2 = dump['y']
    X3 = dump['z']

    dvar_code = []
    dvar_code.append(dump['RHO'] - var0[0])
    dvar_code.append(dump['UU'] - var0[1])
    dvar_code.append(dump['U1'] - var0[2])
    dvar_code.append(dump['U2'] - var0[3])
    dvar_code.append(dump['U3'] - var0[4])
    dvar_code.append(dump['B1'] - var0[5])
    dvar_code.append(dump['B2'] - var0[6])
    dvar_code.append(dump['B3'] - var0[7])

    dvar_sol = []
    L1.append([])
    for k in range(NVAR):
      dvar_sol.append(np.real(dvar[k])*np.cos(k1*X1 + k2*X2 + k3*X3))
      L1[m].append(np.mean(np.fabs(dvar_code[k] - dvar_sol[k])))

# MEASURE CONVERGENCE
L1 = np.array(L1)
powerfits = [0.,]*NVAR
for k in range(NVAR):
    if abs(dvar[k]) != 0.:
        powerfits[k] = np.polyfit(np.log(RES), np.log(L1[:,k]), 1)[0]
        print("Power fit var {}: {}".format(k, powerfits[k]))

# MAKE PLOTS
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(1,1,1)
for k in range(NVAR):
    if abs(dvar[k]) != 0.:
        ax.plot(RES, L1[:,k], marker='s', label=VARS[k])

norm = L1[0,0]*RES[0]*RES[0]
if norm < 1e-4:
    norm = L1[0,3]*RES[0]*RES[0]
xmin = RES[0]/2.
xmax = RES[-1]*2.
ax.plot([xmin, xmax], norm*np.asarray([xmin, xmax])**-2., color='k', linestyle='--', label='N^-2')

plt.xscale('log', base=2); plt.yscale('log')
plt.xlim([RES[0]/np.sqrt(2.), RES[-1]*np.sqrt(2.)])
plt.xlabel('N'); plt.ylabel('L1')
plt.title("MHD mode test convergence, {}".format(LONG))
plt.legend(loc=1)
plt.savefig("convergence_modes_{}.png".format(SHORT))

