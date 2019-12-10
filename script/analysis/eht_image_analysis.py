################################################################################
#                                                                              # 
#  CALCULATE TIME-AVERAGED QUANTITIES FROM IPOLE IMAGES                        #
#                                                                              # 
################################################################################

from __future__ import print_function, division

from analysis_fns import *
import os, glob
import util

import sys
import pickle

import psutil,multiprocessing

import numpy as np

impath = sys.argv[1]
debug = 0

# M87 parameters
Msun = 1.989e33
M = 6.2e9*Msun
G = 6.67428e-8
c = 2.99792e10
pc = 3.08568e18
d = 16.9*1.e6*pc
# size of single pixel in rad: M/pixel . muas/pix . rad/muas
muas_per_M = G*M/(c*c*d) * 1.e6 * 206264.8
M_per_muas = 1./muas_per_M

# pixel size in radians
da = 1. / (1.e6 * 206265.)
# solid angle subtended by pixel
dO = da*da
Jy = 1.e-23  # cgs


# Shamelessly stolen from CFG's 'ipole_plot.py'
# TODO new ipole format too
files = np.sort(glob.glob(os.path.join(impath,"*.dat")))
foldername = os.path.basename(impath)

# Image names store a bunch of info we want to keep around
# FORM: image_a+0.94_1000_163_0_230.e9_6.2e9_7.791e+24_10.dat
# Or something like it...
# Hackish heuristic detection follows, not for the squeamish

def parse_params(fname):
  fname_split = os.path.basename(fname)[:-4].split("_")
  ints = []
  floats = []
  for bit in fname_split:
    try:
      if len(bit) != 4:
        ints.append(int(bit))
    except ValueError as e:
      pass
    try:
      floats.append(float(bit))
    except ValueError as e:
      pass

  params = {}
  params['spin'] = [bit for bit in fname_split if ("a+" in bit or "a-" in bit or bit == "a0")]
  params['angle'] = [bit for bit in ints if bit in [158,163,168,12,17,22]]
  params['freq'] = [bit for bit in floats if bit > 100.e9 and bit < 1000.e9]
  params['mass'] = [bit for bit in floats if bit > 1.e9 and bit < 10.e9]
  params['munit'] = [bit for bit in floats if bit > 1.e20 and bit < 1.e50]
  params['rhigh'] = [bit for bit in ints if bit in [1,10,20,40,80,160]]

  if len(params['rhigh']) == 2 and (1 in params['rhigh']):
    params['rhigh'].remove(1)

  for key in params:
    if len(params[key]) > 1:
      print("Failed to parse fileaname!")
      print("Parameter {} has values {} for file {}!".format(key, params[key], os.path.basename(files[0])))
      exit(-1)
    elif len(params[key]) == 0:
      if key == "rhigh":
        params['rhigh'] = None
      else:
        print("Param {} not present in filename {}".format(key, os.path.basename(files[0])))
    else:
      params[key] = params[key][0]

  return(params)

params_global = parse_params(files[0])

# Make sure we get the low-angle runs of negative spins
n = 0
if "a-" in params_global['spin']:
  while params_global['angle'] not in [12,17,22]:
    n += 1
    params_global = parse_params(files[n])

global_param_n = n

#print("Run parameters: fname={}, spin={}, angle={}, freq={}, mass={}, munit={}, rhigh={}".format(
#       fname, spin, angle, freq, mass, munit, rhigh))

def process(n):
  # Skip file if it wasn't the same run
  if parse_params(files[n]) != params_global:
    print("File {} is from different run than {}. Skipping.".format(files[n],files[global_param_n]))
    return None

  # read in data
  i0, j0, Ia, Is, Qs, Us, Vs = np.loadtxt(files[n], unpack=True)
  print("Read {} / {}".format(n,len(files)))

  out = {}
  # Keep full images to average them into another
  out['i0'] = i0
  out['j0'] = j0
  out['Ia'] = Ia
  out['Is'] = Is
  out['Qs'] = Qs
  out['Us'] = Us
  out['Vs'] = Vs

  # set image size: assumed square!
  out['ImRes'] = ImRes = int(round(np.sqrt(len(i0))))

  out['FOV'] = ImRes*M_per_muas

  out['flux_pol'] = dO*sum(Is)/Jy
  out['flux_unpol'] = dO*sum(Ia)/Jy

  out['I_sum'] = Ib = sum(Is)
  out['Q_sum'] = Qb = sum(Qs)
  out['U_sum'] = Ub = sum(Us)
  out['V_sum'] = Vb = sum(Vs)

  out['LP_frac'] = np.sqrt(Qb*Qb + Ub*Ub)/Ib
  out['CHI'] = (180./3.14159)*0.5*np.arctan2(Ub,Qb)
  out['CP_frac'] = Vb/Ib
  #TODO EVPA?

  return out

if __name__ == "__main__":
  if debug:
    # SERIAL (very slow)
    out_list = [process(n) for n in range(len(files))]
  else:
    # PARALLEL
    #NTHREADS = util.calc_nthreads(hdr, pad=0.3)
    NTHREADS = psutil.cpu_count(logical=False)
    pool = multiprocessing.Pool(NTHREADS)
    try:
      # Map the above function to the dump numbers, returning a list of 'out' dicts
      out_list = pool.map_async(process, list(range(len(files)))).get(99999999)
      #print out_list[0].keys()
    except KeyboardInterrupt:
      pool.terminate()
      pool.join()
    else:
      pool.close()
      pool.join()

  out_list = [x for x in out_list if x is not None]

  ND = len(out_list)
  out_full = {}
  for key in out_list[0].keys():
    if key in ['i0', 'j0', 'Ia', 'Is', 'Qs', 'Us', 'Vs']:
      # Average the image parameter keys
      out_full[key] = np.zeros_like(out_list[0][key])
      for n in range(ND):
        out_full[key] += out_list[n][key]
      out_full[key] /= ND
    else:
      # Record all the individual number keys
      out_full[key] = np.zeros(ND)
      for n in range(ND):
        out_full[key][n] = out_list[n][key]

  for key in out_full:
    if key not in ['i0', 'j0', 'Ia', 'Is', 'Qs', 'Us', 'Vs']:
      print("Average {} is {}".format(key, np.mean(out_full[key])))

  # Output average image
  cols_array = np.c_[out_full['i0'], out_full['j0'], out_full['Ia'], out_full['Is'], out_full['Qs'], out_full['Us'], out_full['Vs']]
  datfile = open("avg_img_{}.dat".format(foldername), "w")
  for i in range(out_full['i0'].size):
    datfile.write("{:.0f} {:.0f} {:g} {:g} {:g} {:g} {:g}\n".format(*cols_array[i]))
  datfile.close()

  # Add params too
  out_full.update(params_global)
  # Tag output with model to avoid writing more bash code
  pickle.dump(out_full, open("im_avgs_{}.p".format(foldername), "wb"))
