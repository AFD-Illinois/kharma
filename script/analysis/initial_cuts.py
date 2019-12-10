## Initial conditions cuts

from __future__ import print_function, division

import hdf5_to_dict as io

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import *

COMPARE = False

dump_dir = sys.argv[1]
init_file = io.get_dumps_list(dump_dir)[0]
hdr, geom, dump = io.load_all(init_file, extras=False)

N2 = hdr['n2']
r = geom['r'][:, N2//2, 0]
rho = dump['RHO'][:, N2//2, 0]
uu = dump['UU'][:, N2//2, 0]
p = (hdr['gam']-1)*uu
b2 = dump['bsq'][:, N2//2, 0]
beta = dump['beta'][:, N2//2, 0]
gamma = dump['gamma'][:, N2//2, 0]

figname = 'initial-cuts.pdf'

if COMPARE:
  tablename = 'initial-cuts.csv'
  data=loadtxt('torus_cuts.csv')
  #data1=loadtxt('data_d2_x+0.16D+01_n0000.csv',skiprows=1,delimiter=',')

r_=0
rho_=1
p_=2
lfac_=4
b2_=3

def betainv(data):
    return data[:,b2_]/2./data[:,p_]
    
f, all_axes = plt.subplots(2, 3, sharex='col')
((ax1, ax2, ax3), (ax4, ax5, ax6)) = all_axes
f.subplots_adjust(wspace=.5)
f.set_size_inches(10,4)

if COMPARE:
  ax1.plot(data[:,r_],data[:,rho_],'r-')
  ax2.plot(data[:,r_],data[:,p_],'r-')
  ax3.plot(data[:,r_],sqrt(data[:,b2_]),'r-')
  ax4.plot(data[:,r_],betainv(data),'r-')
  ax5.plot(data[:,r_],data[:,lfac_],'r-')
  ax6.plot(data[:,r_],data[:,p_]+data[:,b2_]/2.,'r-')


ax1.plot(r,rho,'b')
ax1.set_ylabel(r'$\rho$')
ax1.set_ylim(1e-8,1)

ax2.plot(r,p,'b')
ax2.set_ylabel(r'$P_{\rm gas}$')
ax2.set_ylim(1e-12,0.2)

ax3.plot(r,sqrt(b2),'b')
ax3.set_ylabel(r'$\sqrt{b_\mu b^\mu}$')
ax3.set_ylim(1.e-4,1.e-2)

ax4.plot(r,1/beta,'b')
ax4.set_ylabel(r'$\beta^{-1}$')
ax4.set_xlabel(r'$r_{\rm KS} [GM/c^2]$')
ax4.set_ylim(1.e-7,1.e-1)

ax5.plot(r,gamma,'b')
ax5.set_ylabel(r'$\Gamma$')
ax5.set_xlabel(r'$r_{\rm KS} [GM/c^2]$')
ax5.set_ylim(0.98,1.25)

ax6.plot(r,(p + b2/2.),'b')
ax6.set_ylabel(r'$P_{\rm gas}+P_{\rm mag}$')
ax6.set_xlabel(r'$r_{\rm KS} [GM/c^2]$')
ax6.set_ylim(1e-12,0.01)

for ax in all_axes.flatten():
  ax.grid(True)
  ax.set_yscale('log')
  ax.set_xlim(2,50)

f.savefig(figname,bbox_inches='tight')
close()

#ascii.write(data[:,[r_,rho_,p_,lfac_,b2_]],tablename,delimiter=',',names=['r','rho','p','lfac','balphabalpha'])
