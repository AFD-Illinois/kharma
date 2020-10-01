"""
 File: plot_convergence_bondi.py
 
 BSD 3-Clause License
 
 Copyright (c) 2020, AFD Group at UIUC
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
 
 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

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

