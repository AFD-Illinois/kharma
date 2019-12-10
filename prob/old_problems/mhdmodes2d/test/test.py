################################################################################
#                                                                              #
# SOD SHOCKTUBE                                                                #
#                                                                              #
################################################################################

import os
import sys; sys.dont_write_bytecode = True
from subprocess import call
from shutil import copyfile
import glob
import numpy as np
#import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as pl

sys.path.insert(0, '../../../script/')
sys.path.insert(0, '../../../script/analysis/')
import util
import hdf5_to_dict as io

AUTO = False
for arg in sys.argv:
  if arg == '-auto':
    AUTO = True

RES = [16, 32, 64, 128]

# LOOP OVER EIGENMODES
MODES = [0,1,2,3] # 1,2,3
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
var0[5] = 1.
L1 = np.zeros([len(MODES), len(RES), NVAR])
powerfits = np.zeros([len(MODES), NVAR])

for n in xrange(len(MODES)):

  # EIGENMODES
  dvar = np.zeros(NVAR)
  if MODES[n] == 0: # ENTROPY
    dvar[0] = 1.
  if MODES[n] == 1: # SLOW/SOUND
    dvar[0] = 0.558104461559
    dvar[1] = 0.744139282078
    dvar[2] = -0.277124827421
    dvar[3] = 0.0630348927707
    dvar[5] = -0.164323721928
    dvar[6] = 0.164323721928
  if MODES[n] == 2: # ALFVEN
    dvar[4] = 0.480384461415
    dvar[7] = 0.877058019307
  if MODES[n] == 3: # FAST
    dvar[0] = 0.476395427447
    dvar[1] = 0.635193903263
    dvar[2] = -0.102965815319
    dvar[3] = -0.316873207561
    dvar[5] = 0.359559114174
    dvar[6] = -0.359559114174
  dvar *= amp
  
  # USE DUMPS IN FOLDERS OF GIVEN FORMAT
  for m in xrange(len(RES)):
    print '../dumps_' + str(RES[m]) + '_' + str(MODES[n])
    os.chdir('../dumps_' + str(RES[m]) + '_' + str(MODES[n]))

    dfile = np.sort(glob.glob('dump*.h5'))[-1]
    
    hdr = io.load_hdr(dfile)
    geom = io.load_geom(hdr, dfile)
    dump = io.load_dump(hdr, geom, dfile) 
    
    X1 = dump['x'][:,:,0]
    X2 = dump['y'][:,:,0]
    dvar_code = []
    dvar_code.append(dump['RHO'][:,:,0] - var0[0]) 
    dvar_code.append(dump['UU'][:,:,0]  - var0[1])
    dvar_code.append(dump['U1'][:,:,0]  - var0[2])
    dvar_code.append(dump['U2'][:,:,0]  - var0[3])
    dvar_code.append(dump['U3'][:,:,0]  - var0[4])
    dvar_code.append(dump['B1'][:,:,0]  - var0[5])
    dvar_code.append(dump['B2'][:,:,0]  - var0[6])
    dvar_code.append(dump['B3'][:,:,0]  - var0[7])

    dvar_sol = []
    for k in xrange(NVAR):
      dvar_sol.append(np.real(dvar[k])*np.cos(k1*X1 + k2*X2))
      L1[n][m][k] = np.mean(np.fabs(dvar_code[k] - dvar_sol[k]))
      
    mid = RES[m]/2
    
    # Plot dvar
    fig = plt.figure(figsize=(16.18,10))
    ax = fig.add_subplot(1,1,1)
    for k in xrange(NVAR):
      if abs(dvar[k]) != 0.:
        ax.plot(X1[:,mid], dvar_code[k][:,mid], marker='s', label=VARS[k])
        ax.plot(X1[:,mid], dvar_sol[k][:,mid], marker='s', label=(VARS[k] + " analytic"))
    plt.title(NAMES[MODES[n]] + ' ' + str(RES[m]))
    plt.legend(loc=1)
    plt.savefig('../test/modes_' + NAMES[MODES[n]] + '_' + str(RES[m]) + '.png', bbox_inches='tight')
    
#     fig = plt.figure(figsize=(16.18,10))
#     ax = fig.add_subplot(1,1,1)
#     for k in xrange(NVAR):
#       if abs(dvar[k]) != 0.:
#         ax.plot(X1[:,mid,mid], dvar_sol[k][:,mid,mid], marker='s', label=(VARS[k] + " analytic"))
#     plt.title(NAMES[MODES[n]] + ' ' + str(RES[m]) + ' analytic')
#     plt.legend(loc=1)
#     plt.savefig('../test/modes_' + NAMES[MODES[n]] + '_' + str(RES[m]) + '_ana' + '.png', bbox_inches='tight')

  # MEASURE CONVERGENCE
  for k in xrange(NVAR):
    if abs(dvar[k]) != 0.:
      powerfits[n,k] = np.polyfit(np.log(RES), np.log(L1[n,:,k]), 1)[0]
  
  os.chdir('../test')

  if not AUTO:
    # MAKE PLOTS
    fig = plt.figure(figsize=(16.18,10))

    ax = fig.add_subplot(1,1,1)
    for k in xrange(NVAR):
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

if AUTO:
  data = {}
  data['SOL'] = -2.*np.zeros([len(MODES), NVAR])  
  data['CODE'] = powerfits
  import pickle
  pickle.dump(data, open('data.p', 'wb'))

