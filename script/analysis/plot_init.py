################################################################################
#                                                                              #
#  GENERATE PLOT OF INITIAL CONDITIONS                                         #
#                                                                              #
################################################################################

from __future__ import print_function, division

import plot as bplt
import util
import hdf5_to_dict as io

import os,sys
import numpy as np
import matplotlib.pyplot as plt

NLINES = 20
SIZE = 600

PLOT_EXTRA = True
if PLOT_EXTRA:
  FIGX = 10
  FIGY = 13
  NPLOTSX = 2
  NPLOTSY = 3
else:
  FIGX = 10
  FIGY = 8
  NPLOTSX = 2
  NPLOTSY = 2

imname = "initial_conditions.png"

if sys.argv[1] == '-d':
  debug = True
  path = sys.argv[2]
else:
  debug = False
  path = sys.argv[1]

files = io.get_dumps_list(path)

if len(files) == 0:
    util.warn("INVALID PATH TO DUMP FOLDER")
    sys.exit(1)

hdr, geom, dump = io.load_all(files[0])

# Plot the first dump, specifically init as in Narayan '12
N1 = hdr['n1']; N2 = hdr['n2']; N3 = hdr['n3']

# Zoom in for smaller SANE torii
if SIZE > geom['r'][-1,0,0]:
  SIZE = geom['r'][-1,0,0]

fig = plt.figure(figsize=(FIGX, FIGY))
# Density profile
ax = plt.subplot(NPLOTSY,NPLOTSX,1)
bplt.radial_plot(ax, geom, dump['RHO'], ylabel=r"$\rho$", n2=N2//2, n3=N3//2,
                 rlim=[8, 2*10**3], ylim=[10**(-4), 2], logr=True, logy=True)

# B-flux thru midplane inside radius
#flux = np.sum(dump['B2'][:,N2//2,:]*geom['gdet'][:,N2//2,None]*hdr['dx1']*hdr['dx3'],axis=-1)

flux_in = np.zeros((N1,))
flux_in[0] = np.sum(dump['B2'][0,N2//2,:]*geom['gdet'][0,N2//2,None]*hdr['dx1']*hdr['dx3'])
for n in range(1,N1):
  flux_in[n] = flux_in[n-1] + np.sum(dump['B2'][n,N2//2,:]*geom['gdet'][n,N2//2,None]*hdr['dx1']*hdr['dx3'])

ax = plt.subplot(NPLOTSY,NPLOTSX,2)
bplt.radial_plot(ax, geom, flux_in, ylabel=r"Flux in r", rlim=[0, SIZE])

# Density 2D
ax = plt.subplot(NPLOTSY,NPLOTSX,3)
bplt.plot_xz(ax, geom, np.log10(dump['RHO']),
             vmin=-4, vmax = 0, label=r"$\log_{10}(\rho)$", window=[0,SIZE,-SIZE/2,SIZE/2])

# Beta 2D
ax = plt.subplot(NPLOTSY,NPLOTSX,4)
bplt.plot_xz(ax, geom, np.log10(dump['beta']),
             label=r"$\beta$", cmap='RdBu_r', vmin=1, vmax=4,
             window=[0,SIZE,-SIZE/2,SIZE/2])
bplt.overlay_field(ax, geom, dump, NLINES)

if PLOT_EXTRA:
  ax = plt.subplot(NPLOTSY,NPLOTSX,5)
  bplt.plot_xz(ax, geom, np.log10(dump['UU']),
               vmin=-4, vmax = 0, label=r"$\log_{10}(U)$",
               window=[0,SIZE,-SIZE/2,SIZE/2])
  
  ax = plt.subplot(NPLOTSY,NPLOTSX,6)
  bplt.plot_xz(ax, geom, np.log10(dump['bsq']),
               label=r"$\log_{10}(b^2)$", cmap='RdBu_r', vmin=-8, vmax=2,
               window=[0,SIZE,-SIZE/2,SIZE/2])
  bplt.overlay_field(ax, geom, dump, NLINES)

plt.tight_layout()

plt.savefig(imname, dpi=100)
plt.close(fig)
