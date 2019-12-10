################################################################################
#                                                                              #
#  LUMINOSITY COMPARISON                                                       #
#                                                                              #
################################################################################

import os, sys
import pickle
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import hdf5_to_dict as io
import plot as bplt
from analysis_fns import *
from luminosity_th_study import overlay_rth_contours


USEARRSPACE=False

run_name = sys.argv[1]

if "SANE" in run_name:
  SIZE = 50
  AT_R = 40
else:
  SIZE = 400
  AT_R = 100

window=[0,SIZE/2,0,SIZE]
FIGX = 15
FIGY = 15

dumpfile = os.path.join("/scratch/03002/bprather/pharm_dumps/M87SimulationLibrary/GRMHD",run_name,"dumps/dump_00001500.h5")
hdr,geom,dump = io.load_all(dumpfile)

plotfile = os.path.join("/work/03002/bprather/stampede2/movies",run_name,"eht_out.p")
avg = pickle.load(open(plotfile, "rb"))

fig = plt.figure(figsize=(FIGX, FIGY))
gs = gridspec.GridSpec(2, 2, width_ratios=[1,2])

ax = plt.subplot(gs[0,0])
bplt.plot_xz(ax, geom, np.log10(d_fns['FE_EM'](dump)), arrayspace=USEARRSPACE, average=True, window=window)
ax.set_title(r"$\log_{10}( -{{T_{EM}}^r}_t )$")

bplt.overlay_contours(ax, geom, geom['r'], [AT_R], color='k')
overlay_rth_contours(ax, geom, avg, legend=True)

ax = plt.subplot(gs[1,0])
bplt.plot_xz(ax, geom, np.log10(d_fns['FE'](dump)), arrayspace=USEARRSPACE, average=True, window=window)
ax.set_title(r"$\log_{10}( -{T^r}_t - \rho u^r )$")

bplt.overlay_contours(ax, geom, geom['r'], [AT_R], color='k')
overlay_rth_contours(ax, geom, avg)

# I can rely on this for now
start = int(avg['avg_start'])//5
end = int(avg['avg_end'])//5
# Average over quiescence
mdav = np.mean(np.abs(avg['mdot'][start:end]))

ax = plt.subplot(gs[0,1])
ax.plot(avg['r'], avg['LBZ_bg1_r']/mdav, label=r"$L_{BZ}$ ($\beta\gamma > 1.0$ cut)", color='k')
ax.plot(avg['r'], avg['LBZ_sigma1_r']/mdav, label=r"$L_{BZ}$ ($\sigma$ > 1 cut)", color='xkcd:green')
ax.plot(avg['r'], avg['LBZ_allp_r']/mdav, label=r"$L_{BZ}$ (FE > 0 cut)", color='xkcd:pink')
ax.plot(avg['r'], avg['LBZ_Be_nob0_r']/mdav, label=r"$L_{BZ}$ ($Be > 0.02$ cut)", color='xkcd:red')
ax.plot(avg['r'], avg['LBZ_mu2_r']/mdav, label=r"$L_{BZ}$ ($\mu > 2$ cut)", color='xkcd:blue')

ax.set_title(r"$L_{BZ} / \dot{M} = \int -{{T_{EM}}^r}_t \sqrt{-g} dx^{\theta} dx^{\phi} / \dot{M}$")
ax.set_xlim([0,SIZE])
ax.set_xlabel("$r$ (M)")
ax.axvline(AT_R, color='k')

#maxes = [np.max(ab_av(avg['LBZ_'+tag+'_r'])[hdr['n1']//4:]) for tag in ['sigma1', 'be_nob1', 'be_nob0']]
#mins = [np.min(ab_av(avg['LBZ_'+tag+'_r'])[hdr['n1']//4:]) for tag in ['sigma1', 'be_nob1', 'be_nob0']]
#yhi = max(maxes); ylow = max(min(mins),1e-4*yhi)
#print(yhi, ylow)
#ax.set_ylim([ylow ,yhi])
if "SANE" in run_name:
  ax.set_yscale('log')

ax.legend(loc='upper right')

ax = plt.subplot(gs[1,1])
ax.plot(avg['r'], avg['Lj_bg1_r']/mdav, label=r"$L_{j}$ ($\beta\gamma > 1.0$ cut)", color='k')
ax.plot(avg['r'], avg['Lj_sigma1_r']/mdav, label=r"$L_{j}$ ($\sigma$ > 1 cut)", color='xkcd:green')
ax.plot(avg['r'], avg['Lj_allp_r']/mdav, label=r"$L_{j}$ (FE > 0 cut)", color='xkcd:pink')
ax.plot(avg['r'], avg['Lj_Be_nob0_r']/mdav, label=r"$L_{j}$ ($Be > 0.02$ cut)", color='xkcd:red')
ax.plot(avg['r'], avg['Lj_mu2_r']/mdav, label=r"$L_{j}$ ($\mu > 2$ cut)", color='xkcd:blue')

ax.set_title(r"$L_{tot} / \dot{M} = \int (-{T^r}_t - \rho u^r) \sqrt{-g} dx^{\theta} dx^{\phi} / \dot{M}$")
ax.set_xlim([0,SIZE])
ax.set_xlabel("$r$ (M)")
ax.axvline(AT_R, color='k')

#maxes = [np.max(ab_av(avg['Ltot_'+tag+'_r'])[hdr['n1']//4:]) for tag in  ['sigma1', 'be_nob1', 'be_nob0']]
#mins = [np.min(ab_av(avg['Ltot_'+tag+'_r'])[hdr['n1']//4:]) for tag in  ['sigma1', 'be_nob1', 'be_nob0']]
#yhi = max(maxes); ylow = max(min(mins),1e-4*yhi)
#print(yhi, ylow)
#ax.set_ylim([ylow,yhi])
if "SANE" in run_name:
  ax.set_yscale('log')

ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig(run_name.replace("/","_")+"_L_study.png", dpi=100)
plt.close(fig)
