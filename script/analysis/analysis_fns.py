
# Convenient analysis functions for physical calculations and averages
# Meant to be imported "from analysis_fns import *" for convenience

import numpy as np
import scipy.fftpack as fft

# Define a dict of names, coupled with the functions required to obtain their variables.
# That way, we only need to specify lists and final operations in eht_analysis,
# AND don't need to cart all these things around in memory
d_fns = {'rho': lambda dump: dump['RHO'],
         'bsq': lambda dump: dump['bsq'],
         'sigma': lambda dump: dump['bsq'] / dump['RHO'],
         'U': lambda dump: dump['UU'],
         'u_t': lambda dump: dump['ucov'][:, :, :, 0],
         'u_phi': lambda dump: dump['ucov'][:, :, :, 3],
         'u^phi': lambda dump: dump['ucon'][:, :, :, 3],
         'FM': lambda dump: dump['RHO'] * dump['ucon'][:, :, :, 1],
         'FE': lambda dump: -T_mixed(dump, 1, 0),
         'FE_EM': lambda dump: -TEM_mixed(dump, 1, 0),
         'FE_Fl': lambda dump: -TFl_mixed(dump, 1, 0),
         'FL': lambda dump: T_mixed(dump, 1, 3),
         'FL_EM': lambda dump: TEM_mixed(dump, 1, 3),
         'FL_Fl': lambda dump: TFl_mixed(dump, 1, 3),
         'Be_b': lambda dump: bernoulli(dump, with_B=True),
         'Be_nob': lambda dump: bernoulli(dump, with_B=False),
         'Pg': lambda dump: (dump['hdr']['gam'] - 1.) * dump['UU'],
         'Pb': lambda dump: dump['bsq'] / 2,
         'Ptot': lambda dump: d_fns['Pg'](dump) + d_fns['Pb'](dump),
         'beta': lambda dump: dump['beta'],
         'betainv': lambda dump: 1/dump['beta'],
         'jcon': lambda dump: dump['jcon'],
         # TODO TODO TODO take geom everywhere or nowhere
         'jcov': lambda geom, dump: jcov(geom, dump),
         'jsq': lambda geom, dump: jsq(geom, dump),
         'B': lambda dump: np.sqrt(dump['bsq']),
         'betagamma': lambda dump: np.sqrt((d_fns['FE_EM'](dump) + d_fns['FE_Fl'](dump))/d_fns['FM'](dump) - 1),
         'Theta': lambda dump: (dump['hdr']['gam'] - 1) * dump['UU'] / dump['RHO'],
         'Thetap': lambda dump: (dump['hdr']['gam_p'] - 1) * (dump['UU']) / dump['RHO'],
         'Thetae': lambda dump: (dump['hdr']['gam_e'] - 1) * (dump['UU']) / dump['RHO'],
         'gamma': lambda geom, dump: get_gamma(geom, dump),
         'JE0': lambda dump: T_mixed(dump, 0, 0),
         'JE1': lambda dump: T_mixed(dump, 1, 0),
         'JE2': lambda dump: T_mixed(dump, 2, 0)
         }
         # Additions I'm unsure of or which are useless
         #'rur' : lambda dump: geom['r']*dump['ucon'][:,:,:,1],
         #'mu' : lambda dump: (d_fns['FE'](dump) + d_fns['FM'](dump)) / d_fns['FM'](dump),

## Physics functions ##

# These are separated to make them faster
def T_con(geom, dump, i, j):
  gam = dump['hdr']['gam']
  return ( (dump['RHO'] + gam*dump['UU'] + dump['bsq'])*dump['ucon'][:,:,:,i]*dump['ucon'][:,:,:,j] +
           ((gam-1)*dump['UU'] + dump['bsq']/2)*geom['gcon'][:,:,None,i,j] - dump['bcon'][:,:,:,i]*dump['bcon'][:,:,:,j] )

def T_cov(geom, dump, i, j):
  gam = dump['hdr']['gam']
  return ( (dump['RHO'] + gam*dump['UU'] + dump['bsq'])*dump['ucov'][:,:,:,i]*dump['ucov'][:,:,:,j] +
           ((gam-1)*dump['UU'] + dump['bsq']/2)*geom['gcov'][:,:,None,i,j] - dump['bcov'][:,:,:,i]*dump['bcov'][:,:,:,j] )

def T_mixed(dump, i, j):
  gam = dump['hdr']['gam']
  if i != j:
    return ( (dump['RHO'] + gam*dump['UU'] + dump['bsq'])*dump['ucon'][:,:,:,i]*dump['ucov'][:,:,:,j] +
             - dump['bcon'][:,:,:,i]*dump['bcov'][:,:,:,j] )
  else:
    return ( (dump['RHO'] + gam*dump['UU'] + dump['bsq']) * dump['ucon'][:,:,:,i]*dump['ucov'][:,:,:,j] +
             (gam-1)*dump['UU'] + dump['bsq']/2 - dump['bcon'][:,:,:,i]*dump['bcov'][:,:,:,j] )

def TEM_mixed(dump, i, j):
  if i != j:
    return dump['bsq'][:,:,:]*dump['ucon'][:,:,:,i]*dump['ucov'][:,:,:,j] - dump['bcon'][:,:,:,i]*dump['bcov'][:,:,:,j]
  else:
    return dump['bsq'][:,:,:]*dump['ucon'][:,:,:,i]*dump['ucov'][:,:,:,j] + dump['bsq']/2 - dump['bcon'][:,:,:,i]*dump['bcov'][:,:,:,j]

def TFl_mixed(dump, i, j):
  gam = dump['hdr']['gam']
  if i != j:
    return (dump['RHO'] + dump['hdr']['gam']*dump['UU'])*dump['ucon'][:,:,:,i]*dump['ucov'][:,:,:,j]
  else:
    return (dump['RHO'] + dump['hdr']['gam']*dump['UU'])*dump['ucon'][:,:,:,i]*dump['ucov'][:,:,:,j] + (gam-1)*dump['UU']

# Return the i,j component of contravarient Maxwell tensor
# TODO there's a computationally easier way to do this:
# Pre-populate an antisym ndarray and einsum
# Same below
def Fcon(geom, dump, i, j):
  NDIM = dump['hdr']['n_dim']

  Fconij = np.zeros_like(dump['RHO'])
  if i != j:
    for mu in range(NDIM):
      for nu in range(NDIM):
        Fconij[:, :, :] += _antisym(i, j, mu, nu) * dump['ucov'][:, :, :, mu] * dump['bcov'][:, :, :, nu]

  # Specify we want gdet in the vectors' coordinate system (this matters for KORAL dump files)
  # TODO is normalization correct?
  return Fconij*geom['gdet'][:,:,None]

def Fcov(geom, dump, i, j):
  NDIM = dump['hdr']['n_dim']

  Fcovij = np.zeros_like(dump['RHO'])
  for mu in range(NDIM):
    for nu in range(NDIM):
      Fcovij += Fcon(geom, dump, mu, nu)*geom['gcov'][:,:,None,mu,i]*geom['gcov'][:,:,None,nu,j]
  
  return Fcovij

def bernoulli(dump, with_B=False):
  if with_B:
    return -T_mixed(dump,0,0) / (dump['RHO']*dump['ucon'][:,:,:,0]) - 1
  else:
    return -(1 + dump['hdr']['gam']*dump['UU']/dump['RHO'])*dump['ucov'][:,:,:,0] - 1

# This is in zone metric!
def lower(geom, vec):
  return np.einsum("...i,...ij->...j", vec, geom['gcov'][:,:,None,:,:])

def to_zone_coords(geom, vec):
  return np.einsum("...i,...ij->...j", vec, geom['vec_to_grid'][:,:,None,:,:])

# Compute 4-vectors given fluid state
# Always returns vectors in the _grid_ coordinate system, to simplify analysis
def get_state(hdr, geom, dump, return_gamma=False):
  ucon = np.zeros([hdr['n1'],hdr['n2'],hdr['n3'],hdr['n_dim']])
  ucov = np.zeros_like(ucon)
  bcon = np.zeros_like(ucon)
  bcov = np.zeros_like(ucon)

  # Aliases to make the below more readable
  if geom['mixed_metrics']:
    # Make sure these are in the vector metric if mixing
    gcov = geom['gcov_vec']
    gcon = geom['gcon_vec']
    alpha = geom['lapse_vec']
  else:
    gcov = geom['gcov']
    gcon = geom['gcon']
    alpha = geom['lapse']

  B1 = dump['B1']
  B2 = dump['B2']
  B3 = dump['B3']

  gamma = get_gamma(geom, dump)

  ucon[:,:,:,0] = gamma/(alpha[:,:,None])
  ucon[:,:,:,1] = dump['U1'] - gamma*alpha[:,:,None]*gcon[:,:,None,0,1]
  ucon[:,:,:,2] = dump['U2'] - gamma*alpha[:,:,None]*gcon[:,:,None,0,2]
  ucon[:,:,:,3] = dump['U3'] - gamma*alpha[:,:,None]*gcon[:,:,None,0,3]

  ucov = np.einsum("...i,...ij->...j", ucon, gcov[:,:,None,:,:])
  bcon[:,:,:,0] = B1*ucov[:,:,:,1] + B2*ucov[:,:,:,2] + B3*ucov[:,:,:,3]
  bcon[:,:,:,1] = (B1 + bcon[:,:,:,0]*ucon[:,:,:,1])/ucon[:,:,:,0]
  bcon[:,:,:,2] = (B2 + bcon[:,:,:,0]*ucon[:,:,:,2])/ucon[:,:,:,0]
  bcon[:,:,:,3] = (B3 + bcon[:,:,:,0]*ucon[:,:,:,3])/ucon[:,:,:,0]

  if geom['mixed_metrics']:
    # Convert all 4-vectors to zone coordinates
    ucon = np.einsum("...i,...ij->...j", ucon, geom['vec_to_grid'][:,:,None,:,:])
    ucov = np.einsum("...i,...ij->...j", ucon, geom['gcov'][:,:,None,:,:]) # Lower with _zone_ metric
    bcon = np.einsum("...i,...ij->...j", bcon, geom['vec_to_grid'][:,:,None,:,:])
    bcov = np.einsum("...i,...ij->...j", bcon, geom['gcov'][:,:,None,:,:])
  else:
    # Already have ucov in this case
    bcov = np.einsum("...i,...ij->...j", bcon, gcov[:,:,None,:,:])

  if return_gamma:
    return ucon, ucov, bcon, bcov, gamma
  else:
    return ucon, ucov, bcon, bcov

def get_gamma(geom, dump):
  # Aliases to make the below more readable
  if geom['mixed_metrics']:
    # Make sure this is in the vector metric if mixing
    gcov = geom['gcov_vec']
  else:
    gcov = geom['gcov']

  U1 = dump['U1']
  U2 = dump['U2']
  U3 = dump['U3']

  qsq = (gcov[:,:,None,1,1]*U1**2 + gcov[:,:,None,2,2]*U2**2 +
         gcov[:,:,None,3,3]*U3**2 + 2.*(gcov[:,:,None,1,2]*U1*U2 +
                                        gcov[:,:,None,1,3]*U1*U3 +
                                        gcov[:,:,None,2,3]*U2*U3))
  return np.sqrt(1. + qsq)

def jcov(geom, dump):
  return np.einsum("...i,...ij->...j", dump['jcon'], geom['gcov'][:,:,None,:,:])

def jsq(geom, dump):
  return np.sum(dump['jcon']*jcov(geom, dump), axis=-1)

# Decide where to measure fluxes
def i_of(geom, rcoord):
  i = 0
  while geom['r'][i,geom['n2']//2,0] < rcoord:
    i += 1
  i -= 1
  return i

## Correlation functions/lengths ##

def corr_midplane(geom, var, norm=True, at_i1=None):
  if at_i1 is None:
    at_i1 = range(geom['n1'])

  jmin = geom['n2']//2-1
  jmax = geom['n2']//2+1

  R = np.zeros((len(at_i1), geom['n3']))

  # TODO is there a way to vectorize over R? Also, are we going to average over adjacent r ever?
  for i1 in at_i1:
    # Average over small angle around midplane
    var_phi = np.mean(var[i1, jmin:jmax, :], axis=0)
    # Calculate autocorrelation
    var_phi_normal = (var_phi - np.mean(var_phi))/np.std(var_phi)
    var_corr = fft.ifft(np.abs(fft.fft(var_phi_normal))**2)
    R[i1] = np.real(var_corr)/(var_corr.size)

  if norm:
    normR = R[:,0]
    for k in range(geom['n3']):
      R[:, k] /= normR

  return R

# TODO needs work...
def jnu_inv(nu, Thetae, Ne, B, theta):
  K2 = 2.*Thetae**2
  nuc = EE * B / (2. * np.pi * ME * CL)
  nus = (2./9.) * nuc * Thetae**2 * np.sin(theta)
  j[nu > 1.e12*nus] = 0.
  x = nu/nus
  f = pow( pow(x, 1./2.) + pow(2.,11./12.)*pow(x,1./6.), 2 )
  j = (sqrt(2.) * np.pi * EE**2 * Ne * nus / (3. *CL * K2)) * f * exp(-pow(x,1./3.))
  return j / nu**2

def corr_midplane_direct(geom, var, norm=True):
  jmin = geom['n2']//2-1
  jmax = geom['n2']//2+1
  
  var_norm = np.ones((geom['n1'], 2, geom['n3']))
  # Normalize radii separately
  for i in range(geom['n1']):
    vmean = np.mean(var[i,jmin:jmax,:])
    var_norm[i,:,:] = var[i,jmin:jmax,:] - vmean
  
  R = np.ones((geom['n1'], geom['n3']))
  for k in range(geom['n3']):
    R[:, k] = np.sum(var_norm*np.roll(var_norm, k, axis=-1)*geom['dx3'], axis=(1,2))/2
    

  if norm:
    normR = R[:, 0]
    for k in range(geom['n3']):
      R[:, k] /= normR
  
  return R

def corr_length(R):
  # TODO this can be done with a one-liner, I know it
  lam = np.zeros(R.shape[0])
  for i in range(R.shape[0]):
    k = 0
    while k < R.shape[1] and R[i, k] >= R[i, 0] / np.exp(1):
      k += 1
    lam[i] = k*(2*np.pi/R.shape[1])
  return lam


## Power Spectra ##
def pspec(var, t, window=0.33, half_overlap=False, bin="fib"):
  if not np.any(var[var.size // 2:]):
    return np.zeros_like(var), np.zeros_like(var)

  data = var[var.size // 2:]
  data = data[np.nonzero(data)] - np.mean(data[np.nonzero(data)])

  if window < 1:
    window = int(window * data.size)
  print("FFT window is ", window)

  sample_time = (t[-1] - t[0]) / t.size
  print("Sampling time is {}".format(sample_time))
  out_freq = np.abs(np.fft.fftfreq(window, sample_time))

  if half_overlap:
    # Hanning w/50% overlap
    spacing = (window // 2)
    nsamples = data.size // spacing

    out = np.zeros(window)
    for i in range(nsamples - 1):
      windowed = np.hanning(window) * data[i * spacing:(i + window//spacing) * spacing]
      out += np.abs(np.fft.fft(windowed)) ** 2

    # TODO binning?

    freqs = out_freq

  else:
    # Hamming no overlap, like comparison paper
    nsamples = data.size // window

    for i in range(nsamples):
      windowed = np.hamming(window) * data[i * window:(i + 1) * window]
      pspec = np.abs(fft.fft(windowed)) ** 2

      # Bin data, declare accumulator output when we know its size
      if bin == "fib":
        # Modify pspec, allocate for modified form
        pspec, freqs = fib_bin(pspec, out_freq)

        if i == 0:
          out = np.zeros_like(np.array(pspec))
      else:
        if i == 0:
          out = np.zeros(window)

      out += pspec

  print("PSD using ", nsamples, " segments.")
  out /= nsamples
  out_freq = freqs

  return out, out_freq

def fib_bin(data, freqs):
  # Fibonacci binning.  Why is this a thing.
  j = 0
  fib_a = 1
  fib_b = 1
  pspec = []
  pspec_freq = []
  while j + fib_b < data.size:
    pspec.append(np.mean(data[j:j + fib_b]))
    pspec_freq.append(np.mean(freqs[j:j + fib_b]))
    j = j + fib_b
    fib_c = fib_a + fib_b
    fib_a = fib_b
    fib_b = fib_c

  return np.array(pspec), np.array(pspec_freq)

## Sums and Averages ##
  
# Var must be a 3D array i.e. a grid scalar
# TODO could maybe be made faster with 'where' but also harder to get right
def sum_shell(geom, var, at_zone=None, mask=None):
  integrand = var * geom['gdet'][:, :, None]*geom['dx2']*geom['dx3']
  if mask is not None:
    integrand *= mask

  if at_zone is not None:
    return np.sum(integrand[at_zone,:,:], axis=(0,1))
  else:
    return np.sum(integrand, axis=(1,2))

def sum_plane(geom, var, within=None):
  jmin = geom['n2']//2-1
  jmax = geom['n2']//2+1
  if within is not None:
    return np.sum(var[:within,jmin:jmax,:] * geom['gdet'][:within,jmin:jmax,None]*geom['dx1']*geom['dx3']) / (jmax-jmin)
  else:
    return np.sum(var[:,jmin:jmax,:] * geom['gdet'][:,jmin:jmax,None]*geom['dx1']*geom['dx3']) / (jmax-jmin)

def sum_vol(geom, var, within=None):
  if within is not None:
    return np.sum(var[:within,:,:] * geom['gdet'][:within,:,None]*geom['dx1']*geom['dx2']*geom['dx3'])
  else:
    return np.sum(var * geom['gdet'][:,:,None]*geom['dx1']*geom['dx2']*geom['dx3'])

def eht_vol(geom, var, jmin, jmax, outside=None):
  if outside is not None:
    return np.sum(var[outside:,jmin:jmax,:] * geom['gdet'][outside:,jmin:jmax,None]*geom['dx1']*geom['dx2']*geom['dx3'])
  else:
    return np.sum(var[:,jmin:jmax,:] * geom['gdet'][:,jmin:jmax,None]*geom['dx1']*geom['dx2']*geom['dx3'])

# TODO can I cache the volume instead of passing these?
def get_j_vals(geom):
  THMIN = np.pi/3.
  THMAX = 2.*np.pi/3.
  # Calculate jmin, jmax for EHT radial profiles
  ths = geom['th'][-1,:,0]
  for n in range(len(ths)):
    if ths[n] > THMIN:
      jmin = n
      break
  
  for n in range(len(ths)):
    if ths[n] > THMAX:
      jmax = n
      break

  return jmin, jmax

# TODO can I cache the volume instead of passing these?
def eht_profile(geom, var, jmin, jmax):
  return ( (var[:,jmin:jmax,:] * geom['gdet'][:,jmin:jmax,None]*geom['dx2']*geom['dx3']).sum(axis=(1,2)) /
           ((geom['gdet'][:,jmin:jmax]*geom['dx2']).sum(axis=1)*2*np.pi) )

def theta_av(geom, var, start, zones_to_av=1, use_gdet=False, fold=True):
  # Sum theta from each pole to equator and take overall mean
  N2 = geom['n2']
  if use_gdet:
    return (var[start:start+zones_to_av,:N2//2,:] * geom['gdet'][start:start+zones_to_av,:N2//2,None]*geom['dx1']*geom['dx3'] +
              var[start:start+zones_to_av,:N2//2-1:-1,:] * geom['gdet'][start:start+zones_to_av,:N2//2-1:-1,None]*geom['dx1']*geom['dx3']).sum(axis=(0,2))\
           /((geom['gdet'][start:start+zones_to_av,:N2//2]*geom['dx1']).sum(axis=0)*2*np.pi)
  else:
    if fold:
      return (var[start:start+zones_to_av,:N2//2,:].mean(axis=(0,2)) + var[start:start+zones_to_av,:N2//2-1:-1,:].mean(axis=(0,2))) / 2
    else:
      return var[start:start+zones_to_av,:,:].mean(axis=(0,2))

## Internal functions ##

# Completely antisymmetric 4D symbol
# TODO cache? Is this validation necessary?
def _antisym(a, b, c, d):
  # Check for valid permutation
  if (a < 0 or a > 3): return 100
  if (b < 0 or b > 3): return 100
  if (c < 0 or c > 3): return 100
  if (d < 0 or d > 3): return 100

  # Entries different? 
  if (a == b or a == c or a == d or
          b == c or b == d or c == d):
    return 0

  return _pp([a, b, c, d])

# Due to Norm Hardy; good for general n
def _pp(P):
  v = np.zeros_like(P)

  p = 0
  for j in range(len(P)):
    if (v[j]):
      p += 1
    else:
      x = j
      while True:
        x = P[x]
        v[x] = 1
        if x == j:
          break

  if p % 2 == 0:
    return 1
  else:
    return -1

