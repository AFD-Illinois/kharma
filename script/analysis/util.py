################################################################################
#                                                                              #
#  UTILITY FUNCTIONS                                                           #
#                                                                              #
################################################################################

import subprocess
import glob
import os

import signal
import multiprocessing
import psutil

import numpy as np

# TODO fns to process argv

# Run a function in parallel with Python's multiprocessing
# 'function' must take only a number
def run_parallel(function, nmax, nthreads, debug=False):
  # TODO if debug...
  pool = multiprocessing.Pool(nthreads)
  try:
    pool.map_async(function, list(range(nmax))).get(720000)
  except KeyboardInterrupt:
    print('Caught interrupt!')
    pool.terminate()
    exit(1)
  else:
    pool.close()
  pool.join()

# Run a function in parallel with Python's multiprocessing
# 'function' must take only a number
# 'merge_function' must take the same number plus whatever 'function' outputs, and adds to the dictionary out_dict
def iter_parallel(function, merge_function, out_dict, nmax, nthreads, debug=False):
  # TODO if debug...
  pool = multiprocessing.Pool(nthreads)
  try:
    # Map the above function to the dump numbers, returning an iterator of 'out' dicts to be merged one at a time
    # This avoids keeping the (very large) full pre-average list in memory
    out_iter = pool.imap(function, list(range(nmax)))
    for n,result in enumerate(out_iter):
      merge_function(n, result, out_dict)
  except KeyboardInterrupt:
    pool.terminate()
    pool.join()
  else:
    pool.close()
    pool.join()

# Calculate ideal # threads
# Lower pad values are safer
def calc_nthreads(hdr, n_mkl=8, pad=0.25):
  # Limit threads for 192^3+ problem due to memory
  # Try to add some parallelism w/MKL.  Don't freak if it doesn't work
  try:
    import ctypes
    
    mkl_rt = ctypes.CDLL('libmkl_rt.so')
    mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
    mkl_get_max_threads = mkl_rt.MKL_Get_Max_Threads
    mkl_set_num_threads(n_mkl)
    print("Using {} MKL threads".format(mkl_get_max_threads()))
  except Exception as e:
    print(e)
    
  # Roughly compute memory and leave some generous padding for multiple copies and Python games
  # (N1*N2*N3*8)*(NPRIM + 4*4 + 6) = size of "dump," (N1*N2*N3*8)*(2*4*4 + 6) = size of "geom"
  # TODO get a better model for this, and save memory in general
  ncopies = hdr['n_prim'] + 4*4 + 6
  nproc = int(pad * psutil.virtual_memory().total/(hdr['n1']*hdr['n2']*hdr['n3']*8*ncopies))
  if nproc < 1: nproc = 1
  if nproc > psutil.cpu_count(logical=False): nproc = psutil.cpu_count(logical=False)
  print("Using {} Python processes".format(nproc))
  return nproc

# COLORIZED OUTPUT
class color:
  BOLD    = '\033[1m'
  WARNING = '\033[1;31m'
  BLUE    = '\033[94m'
  NORMAL  = '\033[0m'

def get_files(PATH, NAME):                                                       
  return np.sort(glob.glob(os.path.join(PATH,'') + NAME))

# PRINT ERROR MESSAGE
def warn(mesg):
  print((color.WARNING + "\n  ERROR: " + color.NORMAL + mesg + "\n"))

# APPEND '/' TO PATH IF MISSING
def sanitize_path(path):
  return os.path.join(path, '')

# SEND OUTPUT TO LOG FILE AS WELL AS TERMINAL
def log_output(sys, logfile_name):
  import re
  f = open(logfile_name, 'w')
  class split(object):
    def __init__(self, *files):
      self.files = files
    def write(self, obj):
      n = 0
      ansi_escape = re.compile(r'\x1b[^m]*m')
      for f in self.files:
        if n > 0:
          f.write(ansi_escape.sub('', obj))
        else:
          f.write(obj)
        f.flush()
        n += 1
    def flush(self):
      for f in self.files:
        f.flush()
  sys.stdout = split(sys.stdout, f)
  sys.stderr = split(sys.stderr, f)

# CREATE DIRECTORY
def make_dir(path):
  if not os.path.exists(path):
    os.makedirs(path)

# CALL rm -rf ON RELATIVE PATHS ONLY
def safe_remove(path):
  import sys
  from subprocess import call
  
  # ONLY ALLOW RELATIVE PATHS
  if path[0] == '/':
    warn("DIRECTORY " + path + " IS NOT A RELATIVE PATH! DANGER OF DATA LOSS")
    sys.exit()
  elif os.path.exists(path):
    call(['rm', '-rf', path])

