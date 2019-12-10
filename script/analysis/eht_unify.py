#!/usr/bin/env python3

import os, sys
import pickle
import numpy as np

import hdf5_to_dict as io

avgs = []
for fname in sys.argv[1:-1]:
  print("Loading {}".format(fname))
  avgs.append(pickle.load(open(fname, "rb")))
  avgs[-1]['fname'] = fname

#for avg in avgs:
#  print("Name: {}, contents: {}".format(avg['fname'], avg.keys()))

num_keys = [len(avg.keys()) for avg in avgs]
avg_max_keys = num_keys.index(max(num_keys))

# TODO organize this damn dict.  HDF5?
direct_list = ['fname', 'a', 'gam', 'gam_e', 'gam_p', 'r', 'th', 'th_eh', 'th_bz', 'phi', 'avg_start', 'avg_end', 'avg_w', 't']
keys_to_sum = [key for key in avgs[avg_max_keys].keys() if key not in direct_list]

uni = {}
for key in keys_to_sum:
  uni[key] = np.zeros_like(avgs[avg_max_keys][key])
  for avg in avgs:
    if key in avg:
      # Keep track of averages w/weights, otherwise just sum since everything's time-dependent
      if (key[-2:] == '_r' or key[-3:] == '_th' or key[-4:] == '_hth' or key[-4:] == '_phi' or
          key[-4:] == '_rth' or key[-6:] == '_thphi' or key[-5:] == '_rphi' or key[-4:] == '_pdf'):
        uni[key] += avg[key]*avg['avg_w']
      elif key[-1:] == 't':
        if uni[key].shape[0] < avg[key].shape[0]:
          uni[key] += avg[key][:uni[key].shape[0]]
        else:
          uni[key][:avg[key].shape[0]] += avg[key]
      else:
        if uni[key].size < avg[key].size:
          uni[key] += avg[key][:uni[key].size]
        else:
          uni[key][:avg[key].size] += avg[key]

for key in direct_list:
  if key in avgs[avg_max_keys].keys():
    uni[key] = avgs[avg_max_keys][key]

# Add compat/completeness stuff
uni['mdot'] = uni['Mdot']
uni['phi_b'] = uni['Phi_b']/np.sqrt(uni['Mdot'])

# Add the log versions of variables, for completeness/better ffts
if os.path.exists(sys.argv[-1]):
  uni['diags'] = io.load_log(sys.argv[-1])

with open("eht_out.p", "wb") as outf:
  print("Writing eht_out.p")
  pickle.dump(uni, outf)
