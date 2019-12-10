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
sys.path.insert(0, '../script/')
sys.path.insert(0, '../script/analysis/')
import util
import hdf5_to_dict as io
TMP_DIR = 'TMP'
TMP_BUILD = 'build_tmp.py'
util.safe_remove(TMP_DIR)

AUTO = False
for arg in sys.argv:
  if arg == '-auto':
    AUTO = True

RES = [16, 32, 64]#, 128]

util.make_dir(TMP_DIR)
os.chdir('../prob/mhdmodes2d/')

copyfile('build.py', TMP_BUILD)
# COMPILE CODE AT MULTIPLE RESOLUTIONS USING SEPARATE BUILD FILE

for n in xrange(len(RES)):
  util.change_cparm('N1TOT', RES[n], TMP_BUILD)
  util.change_cparm('N2TOT', RES[n], TMP_BUILD)
  call(['python', TMP_BUILD, '-dir', TMP_DIR])
  call(['cp', os.path.join(os.getcwd(), TMP_DIR, 'bhlight'),
        '../../test/' + TMP_DIR + '/bhlight_' + str(RES[n])])
copyfile(os.path.join(os.getcwd(), TMP_DIR, 'param_template.dat'), '../../test/' + 
         TMP_DIR + '/param_template.dat')
util.safe_remove(TMP_BUILD)
util.safe_remove(TMP_DIR)
os.chdir('../../test/')

# LOOP OVER EIGENMODES
MODES = [1, 2, 3]
NAMES = ['ENTROPY', 'SLOW', 'ALFVEN', 'FAST']
NVAR = 8
VARS = ['rho', 'u', 'u1', 'u2', 'u3', 'B1', 'B2', 'B3']

amp = 1.e-4
k1 = 2.*np.pi
k2 = 2.*np.pi
var0 = np.zeros(NVAR)
var0[0] = 1.
var0[1] = 1.
var0[5] = 1.
L1 = np.zeros([len(MODES), len(RES), NVAR])
powerfits = np.zeros([len(MODES), NVAR])

for n in xrange(len(MODES)):
  util.change_rparm('nmode', MODES[n], TMP_DIR + '/param_template.dat')
  os.chdir(TMP_DIR)
  print os.getcwd()

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
  
  # RUN PROBLEM FOR EACH RESOLUTION AND ANALYZE RESULT
  for m in xrange(len(RES)):
    print ['./bhlight_' + str(RES[m]), '-p', 'param_template.dat']
    call(['./bhlight_' + str(RES[m]), '-p', 'param_template.dat'])

    dfiles = np.sort(glob.glob('dumps/dump*.h5'))
    dump = io.load_dump(dfiles[-1]) 
    X1 = dump['X1'][:,:,0]
    X2 = dump['X2'][:,:,0]
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
      if abs(dvar[k]) != 0.:
        L1[n][m][k] = np.mean(np.fabs(dvar_code[k] - dvar_sol[k]))

  # MEASURE CONVERGENCE
  for k in xrange(NVAR):
    if abs(dvar[k]) != 0.:
      powerfits[n,k] = np.polyfit(np.log(RES), np.log(L1[n,:,k]), 1)[0]
  
  os.chdir('../')

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
    plt.savefig('mhdmodes2d_' + NAMES[MODES[n]] + '.png', bbox_inches='tight')

if AUTO:
  data = {}
  data['SOL'] = -2.*np.zeros([len(MODES), NVAR])  
  data['CODE'] = powerfits
  import pickle
  pickle.dump(data, open('data.p', 'wb'))

# CLEAN UP
util.safe_remove(TMP_DIR)

