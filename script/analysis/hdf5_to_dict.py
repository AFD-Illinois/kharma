################################################################################
#                                                                              # 
#  READ HARM OUTPUT                                                            #
#                                                                              # 
################################################################################

from __future__ import print_function, division

import os, sys
from pkg_resources import parse_version

import numpy as np
import h5py
import glob

import units
from analysis_fns import *

# New infra
from defs import Loci, Met
from coordinates import dxdX_to_KS, dxdX_KS_to

class HARMdump(object):
  def __init__(self, dfname):
    self.dfile = h5py.File(dfname)
  def __getitem__(self, name):
    return d_fns[name](self.dfile)
  def __del__(self):
    self.dfile.close()

def get_dumps_list(path):
  # Funny how many names output has
  files_harm = [file for file in glob.glob(os.path.join(path,"*dump*.h5"))]
  files_koral = [file for file in glob.glob(os.path.join(path,"*sim*.h5"))]
  files_bhac = [file for file in glob.glob(os.path.join(path,"*data*.h5"))]
  return np.sort(files_harm + files_koral + files_bhac)

def get_full_dumps_list(path):
  alldumps = get_dumps_list(path)
  fulldumps = []

  for fname in alldumps:
    dfile = h5py.File(fname, 'r')
    if dfile['is_full_dump'][()] == 1:
      fulldumps.append(fname)
    dfile.close()
  return np.sort(fulldumps)

# For single plotting scripts
def load_all(fname, **kwargs):
  hdr = load_hdr(fname)
  path = os.path.dirname(fname)
  geom = load_geom(hdr, path)
  dump = load_dump(fname, hdr, geom, **kwargs)
  return hdr, geom, dump

# For cutting on time without loading everything
def get_dump_time(fname):
  dfile = h5py.File(fname, 'r')

  if 't' in dfile.keys():
    t = dfile['t'][()]
  else:
    t = 0

  dfile.close()
  return t

# Function to recursively un-bytes all the dumb HDF5 strings
def decode_all(dict):
    for key in dict:
      # Decode bytes
      if type(dict[key]) == np.bytes_:
        dict[key] = dict[key].decode('UTF-8')
      # Split ndarray of bytes into list of strings
      elif type(dict[key]) == np.ndarray:
        if dict[key].dtype.kind == 'S':
          dict[key] = [el.decode('UTF-8') for el in dict[key]]
      # Recurse for any subfolders
      elif type(dict[key]) in [list, dict]:
        decode_all(dict[key])

def load_hdr(fname):
  dfile = h5py.File(fname, 'r')

  hdr = {}
  try:
    # Scoop all the keys that are not folders
    for key in [key for key in list(dfile['header'].keys()) if not key == 'geom']:
      hdr[key] = dfile['header/' + key][()]
      
    # TODO load these from grid.h5? Or is the header actually the place for them?
    for key in [key for key in list(dfile['header/geom'].keys()) if not key in ['mks', 'mmks', 'mks3'] ]:
      hdr[key] = dfile['header/geom/' + key][()]
    # TODO there must be a shorter/more compat way to do the following
    if 'mks' in list(dfile['header/geom'].keys()):
      for key in dfile['header/geom/mks']:
        hdr[key] = dfile['header/geom/mks/' + key][()]
    if 'mmks' in list(dfile['header/geom'].keys()):
      for key in dfile['header/geom/mmks']:
        hdr[key] = dfile['header/geom/mmks/' + key][()]
    if 'mks3' in list(dfile['header/geom'].keys()):
      for key in dfile['header/geom/mks3']:
        hdr[key] = dfile['header/geom/mks3/' + key][()]

  except KeyError as e:
    util.warn("File is older than supported by this library. Use hdf5_to_dict_old.py")
    exit(-1)

  decode_all(hdr)

  # Turn the version string into components
  if 'version' not in hdr.keys():
    hdr['version'] = "iharm-alpha-3.6"
    print("Unknown version: defaulting to {}".format(hdr['version']))

  hdr['codename'], hdr['codestatus'], hdr['vnum'] = hdr['version'].split("-")
  hdr['vnum'] = [int(x) for x in hdr['vnum'].split(".")]

  # HARM-specific workarounds:
  if hdr['codename'] == "iharm":
    # Work around naming bug before output v3.4
    if hdr['vnum'] < [3,4]:
      names = []
      for name in hdr['prim_names'][0]:
        names.append( name )
      hdr['prim_names'] = names
    
    # Work around bad radius names before output v3.6
    if ('r_in' not in hdr) and ('Rin' in hdr):
      hdr['r_in'], hdr['r_out'] = hdr['Rin'], hdr['Rout']
    
    # Grab the git revision if that's something we output
    if 'extras' in dfile.keys() and 'git_version' in dfile['extras'].keys():
      hdr['git_version'] = dfile['/extras/git_version'][()].decode('UTF-8')
  
  dfile.close()

  # Patch things that sometimes people forget to put in the header
  if 'n_dim' not in hdr:
    hdr['n_dim'] = 4
  if 'prim_names' not in hdr:
    if hdr['n_prim'] == 10:
      hdr['prim_names'] = ["RHO", "UU", "U1", "U2", "U3", "B1", "B2", "B3", "KEL", "KTOT"]
    else:
      hdr['prim_names'] = ["RHO", "UU", "U1", "U2", "U3", "B1", "B2", "B3"]
  if 'has_electrons' not in hdr:
    if hdr['n_prim'] == 10:
      hdr['has_electrons'] = True
    else:
      hdr['has_electrons'] = False
  # TODO this is KS-specific
  if 'r_eh' not in hdr and hdr['metric'] != "MINKOWSKI":
    hdr['r_eh'] = (1. + np.sqrt(1. - hdr['a']**2))
  if 'poly_norm' not in hdr and hdr['metric'] == "MMKS":
    hdr['poly_norm'] = 0.5 * np.pi * 1. / (1. + 1. / (hdr['poly_alpha'] + 1.) *
                                     1. / np.power(hdr['poly_xt'], hdr['poly_alpha']))
  
  if 'git_version' in hdr:
    print("Loaded header from code {}, git rev {}".format(hdr['version'], hdr['git_version']))
  else:
    print("Loaded header from code {}".format(hdr['version']))

  return hdr

def load_geom(hdr, path):
  # Allow override by making path a filename
  if ".h5" in path:
    fname = path
  else:
    # Otherwise use encoded or default info
    if 'gridfile' in hdr:
      fname = os.path.join(path, hdr['gridfile'])
    else:
      fname = os.path.join(path, "grid.h5")
    
  gfile = h5py.File(fname, 'r')

  geom = {}
  for key in list(gfile['/'].keys()):
    geom[key] = gfile[key][()]

  # Useful stuff for direct access in geom. TODO r_isco if available
  for key in ['n1', 'n2', 'n3', 'dx1', 'dx2', 'dx3', 'startx1', 'startx2', 'startx3', 'n_dim', 'metric']:
    geom[key] = hdr[key]
  if hdr['metric'] in ["MKS", "MMKS", "FMKS"]:
    for key in ['r_eh', 'r_in', 'r_out', 'a', 'hslope']:
      geom[key] = hdr[key]
      if hdr['metric'] == "MMKS": # TODO standardize names !!!
        for key in ['poly_norm', 'poly_alpha', 'poly_xt', 'mks_smooth']:
          geom[key] = hdr[key]
  elif hdr['metric'] in ["MKS3"]:
    for key in ['r_eh']:
      geom[key] = hdr[key]
    geom['r_out'] = geom['r'][-1,hdr['n2']//2,0]

  # these get used interchangeably and I don't care
  geom['x'] = geom['X']
  geom['y'] = geom['Y']
  geom['z'] = geom['Z']

  if 'phi' not in geom and hdr['metric'] in ["MKS", "MMKS", "FMKS", "MKS3"]:
    geom['phi'] = geom['X3']

  # Sometimes the vectors and zones use different coordinate systems
  # TODO allow specifying both systems
  if 'gdet_zone' in geom:
    # Preserve 
    geom['gcon_vec'] = geom['gcon']
    geom['gcov_vec'] = geom['gcov']
    geom['gdet_vec'] = geom['gdet']
    geom['lapse_vec'] = geom['lapse']
    # But default to the grid metric.  Lots of integrals and later manipulation with this
    geom['gcon'] = geom.pop('gcon_zone',None)
    geom['gcov'] = geom.pop('gcov_zone',None)
    geom['gdet'] = geom.pop('gdet_zone',None)
    geom['lapse'] = geom.pop('lapse_zone',None)

    geom['mixed_metrics'] = True
  else:
    geom['mixed_metrics'] = False

  # Compress geom in phi for normal use
  for key in ['gdet', 'lapse', 'gdet_vec', 'lapse_vec']:
    if key in geom:
      geom[key] = geom[key][:,:,0]

  for key in ['gcon', 'gcov', 'gcon_vec', 'gcov_vec']:
    if key in geom:
      geom[key] = geom[key][:,:,0,:,:]
  
  if geom['mixed_metrics']:
    # Get all Kerr-Schild coordinates for generating transformation matrices
    Xgeom = np.zeros((4,geom['n1'],geom['n2']))
    Xgeom[1] = geom['r'][:,:,0]
    Xgeom[2] = geom['th'][:,:,0]
    # TODO add all metric params to the geom dict
    eks2ks = dxdX_to_KS(Xgeom, Met.EKS, hdr, koral_rad=hdr['has_electrons'])
    ks2mks3 = dxdX_KS_to(Xgeom, Met[geom['metric']], hdr, koral_rad=hdr['has_electrons'])
    print("Will convert vectors in EKS to zone metric {}".format(geom['metric']))
    geom['vec_to_grid'] = np.einsum("ij...,jk...->...ik", eks2ks, ks2mks3)

  return geom

def load_dump(fname, hdr, geom, derived_vars=True, extras=True):
  dfile = h5py.File(fname, 'r')
  
  dump = {}
  
  # Carry pointers to header. Saves some pain getting shapes/parameters for plots
  # Geometry, however, _must be carried separately_ due to size in memory
  dump['hdr'] = hdr

  # TODO this necessarily grabs the /whole/ primitives array
  for key in [key for key in list(dfile['/'].keys()) if key not in ['header', 'extras', 'prims'] ]:
    dump[key] = dfile[key][()]

  # TODO should probably error at this one
  if 't' not in dump:
    dump['t'] = 0.

  for name, num in zip(hdr['prim_names'], list(range(hdr['n_prim']))):
    dump[name] = dfile['prims'][:,:,:,num]

  if extras and 'extras' in dfile.keys():
    # Load the extras.
    for key in list(dfile['extras'].keys()):
      dump[key] = dfile['extras/' + key][()]
  
  dfile.close()

  # Recalculate all the derived variables, if we need to
  if derived_vars:
    dump['ucon'], dump['ucov'], dump['bcon'], dump['bcov'] = get_state(hdr, geom, dump)
    dump['bsq'] = (dump['bcon']*dump['bcov']).sum(axis=-1)
    dump['beta'] = 2.*(hdr['gam']-1.)*dump['UU']/(dump['bsq'])

    if hdr['has_electrons']:
      ref = units.get_cgs()
      dump['Thetae'] = ref['MP']/ref['ME']*dump['KEL']*dump['RHO']**(hdr['gam_e']-1.)
      dump['ue'] = dump['KEL']*dump['RHO']**(hdr['gam_e']) / (hdr['gam_e']-1.)
      dump['up'] = dump['UU'] - dump['ue']
      dump['TpTe'] = (hdr['gam_p']-1.)*dump['up']/((hdr['gam_e']-1.)*dump['ue'])

  return dump

def load_log(path):
  # TODO specify log name in dumps, like grid
  logfname = os.path.join(path,"log.out")
  if not os.path.exists(logfname):
    return None
  dfile = np.loadtxt(logfname).transpose()
  
  # TODO log should probably have a header
  diag = {}
  diag['t'] = dfile[0]
  diag['rmed'] = dfile[1]
  diag['pp'] = dfile[2]
  diag['e'] = dfile[3]
  diag['uu_rho_gam_cent'] = dfile[4]
  diag['uu_cent'] = dfile[5]
  diag['mdot'] = dfile[6]
  diag['edot'] = dfile[7]
  diag['ldot'] = dfile[8]
  diag['mass'] = dfile[9]
  diag['egas'] = dfile[10]
  diag['Phi'] = dfile[11]
  diag['phi'] = dfile[12]
  diag['jet_EM_flux'] = dfile[13]
  diag['divbmax'] = dfile[14]
  diag['lum_eht'] = dfile[15]
  diag['mdot_eh'] = dfile[16]
  diag['edot_eh'] = dfile[17]
  diag['ldot_eh'] = dfile[18]

  return diag

# For adding contents of the log to dumps
def log_time(diag, var, t):
  if len(diag['t'].shape) < 1:
    return diag[var]
  else:
    i = 0
    while i < len(diag['t']) and diag['t'][i] < t:
      i += 1
    return diag[var][i-1]
