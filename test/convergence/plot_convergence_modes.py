"""
 File: plot_convergence_modes.py
 
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

################################################################################
#                                                                              #
# MHD MODES CONVERGENCE PLOTS                                                  #
#                                                                              #
################################################################################

from __future__ import print_function, division

import plot as bplt
import util
import hdf5_to_dict as io

import os,sys
import numpy as np
import matplotlib.pyplot as plt

RES = [16,32,64] #,128]

# LOOP OVER EIGENMODES
MODES = [1,2,3]
NAMES = ['ENTROPY', 'SLOW', 'ALFVEN', 'FAST']
NVAR = 8
VARS = ['rho', 'u', 'u1', 'u2', 'u3', 'B1', 'B2', 'B3']

amp = 1.e-4
k1 = 2.*np.pi
k2 = 2.*np.pi
k3 = 2.*np.pi
var0 = np.zeros(NVAR)
var0[0] = 1.
var0[1] = 1.

# Magnetic field
var0[5] = 1.
var0[6] = 0.
var0[7] = 0.

L1 = np.zeros([len(MODES), len(RES), NVAR])
powerfits = np.zeros([len(MODES), NVAR])

for n in range(len(MODES)):

  # EIGENMODES
  dvar = np.zeros(NVAR)
  if MODES[n] == 0: # ENTROPY
    dvar[0] = 1.
  if MODES[n] == 1: # SLOW/SOUND
    dvar[0] = 0.556500332363
    dvar[1] = 0.742000443151
    dvar[2] = -0.282334999306
    dvar[3] = 0.0367010491491
    dvar[4] = 0.0367010491491
    dvar[5] = -0.195509141461
    dvar[6] = 0.0977545707307
    dvar[7] = 0.0977545707307
  if MODES[n] == 2: # ALFVEN
    dvar[3] =  -0.339683110243
    dvar[4] =  0.339683110243
    dvar[6] =  0.620173672946
    dvar[7] =  -0.620173672946
  if MODES[n] == 3: # FAST
    dvar[0]  =  0.481846076323;
    dvar[1]    =  0.642461435098;
    dvar[2]   =  -0.0832240462505;
    dvar[3]   =  -0.224080007379;
    dvar[4]   =  -0.224080007379;
    dvar[5]   =  0.406380545676;
    dvar[6]   =  -0.203190272838;
    dvar[7]   =  -0.203190272838;
  dvar *= amp

  # USE DUMPS IN FOLDERS OF GIVEN FORMAT
  for m in range(len(RES)):
    os.chdir('../dumps_' + str(RES[m]) + '_' + str(MODES[n]))

    dfile = io.get_dumps_list(".")[-1]

    hdr, geom, dump = io.load_all(dfile)

    X1 = geom['x']
    X2 = geom['y']
    X3 = geom['z']

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
    for k in range(NVAR):
      dvar_sol.append(np.real(dvar[k])*np.cos(k1*X1 + k2*X2 + k3*X3))
      L1[n][m][k] = np.mean(np.fabs(dvar_code[k] - dvar_sol[k]))

    mid = RES[m]/2

  # MEASURE CONVERGENCE
  for k in range(NVAR):
    if abs(dvar[k]) != 0.:
      powerfits[n,k] = np.polyfit(np.log(RES), np.log(L1[n,:,k]), 1)[0]

  os.chdir('../plots')

  # MAKE PLOTS
  fig = plt.figure(figsize=(16.18,10))

  ax = fig.add_subplot(1,1,1)
  for k in range(NVAR):
    if abs(dvar[k]) != 0.:
      ax.plot(RES, L1[n,:,k], marker='s', label=VARS[k])

  ax.plot([RES[0]/2., RES[-1]*2.],
    10.*amp*np.asarray([RES[0]/2., RES[-1]*2.])**-2.,
    color='k', linestyle='--', label='N^-2')
  plt.xscale('log', basex=2); plt.yscale('log')
  plt.xlim([RES[0]/np.sqrt(2.), RES[-1]*np.sqrt(2.)])
  plt.xlabel('N'); plt.ylabel('L1')
  plt.title(NAMES[MODES[n]])
  plt.legend(loc=1)
  plt.savefig('mhdmodes3d_' + NAMES[MODES[n]] + '.png', bbox_inches='tight')

