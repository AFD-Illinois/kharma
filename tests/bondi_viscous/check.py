#!/usr/bin/env python3

import os, sys

import numpy as np
from scipy.interpolate import splrep
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt

import pyharm
import pyharm.grmhd.bondi as bondi
import pyharm.plots.plot_dumps as pplt

# Check that the computed Bondi solution matches
# the analytic Bondi solution in rho,u and the
# ODE results in dP

if __name__ == '__main__':
    outputdir = './'
    kharmadir = '../../'

    NVAR  = 3
    VARS  = ['rho', 'u', 'dP']
    RES   = [int(r) for r in sys.argv[1].split(",")]
    LONG  = sys.argv[2]
    SHORT = sys.argv[3]

    L1  = np.zeros([len(RES), NVAR])
    fit = np.zeros([len(RES), NVAR])

    for r, res in enumerate(RES):

        # Load dump for parameters
        dump = pyharm.load_dump("emhd_2d_{}_end_emhd2d_weno.phdf".format(res), cache_conn=True)

        # Compute analytic reference
        mdot, rc, gam = dump['bondi/mdot'], dump['bondi/rs'], dump['gam']
        eta, tau = dump['emhd/eta'], dump['emhd/tau']
        state = bondi.get_bondi_fluid_state(mdot, rc, gam, dump.grid)
        state.params['eta'] = eta
        state.params['tau'] = tau
        dP_check = bondi.compute_dP(mdot, rc, gam, dump.grid, eta, tau)

        # load code data
        dump = pyharm.load_dump("emhd_2d_{}_end_emhd2d_weno.phdf".format(res))

        rho, uu, dP_tilde = dump['RHO'], dump['UU'], dump['dP']
        #rho, uu = dump['RHO'], dump['UU']

        # compute dP
        if dump['emhd/higher_order_terms'] == "true":
            print("Res: "+str(res)+"; higher order terms enabled")
            Theta    = (dump['gam'] - 1.) * uu / rho
            nu_emhd  = eta / rho
            dP       = dP_tilde * np.sqrt(nu_emhd * rho * Theta / tau)
        else:
            dP = dP_tilde

        # Plot
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1,1,1)
        pplt.plot_diff_xz(ax, dump, state, 'rho')
        plt.legend()
        fig.savefig("compare_rho_{}.png".format(res))
        plt.close(fig)

        # compute L1 norm
        L1[r,0] = np.mean(np.fabs(rho[:,0,0] - state['rho'][:,0,0]))
        L1[r,1] = np.mean(np.fabs(uu[:,0,0]  - state['u'][:,0,0]))
        L1[r,2] = np.mean(np.fabs(dP[:,0,0]  - dP_check)[1:-1])

    # MEASURE CONVERGENCE
    L1 = np.array(L1)
    powerfits = [0.,]*NVAR
    fail = 0
    for k in range(NVAR):
        powerfits[k] = np.polyfit(np.log(RES), np.log(L1[:,k]), 1)[0]
        print("Power fit {}: {} {}".format(VARS[k], powerfits[k], L1[:,k]))
        if powerfits[k] > -2 or powerfits[k] < -2.7:
            fail = 1
            
            
    # plotting parameters
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['figure.autolayout'] = True
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['axes.xmargin'] = 0.02
    mpl.rcParams['axes.ymargin'] = 0.02
    mpl.rcParams['legend.fontsize'] = 'medium'
    colors = ['indigo', 'goldenrod', 'darkgreen', 'crimson', 'xkcd:blue']


    # plot
    plt.close()
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)

    # loop over prims
    tracker = 0
    for n in range(len(VARS)):
        color = colors[tracker]
        ax.loglog(RES, L1[:,n], color=color, marker='o', label=VARS[n])
        tracker+=1

    ax.loglog([RES[0], RES[-1]], 0.1*np.asarray([float(RES[0]), float(RES[-1])])**(-2), color='k', linestyle='dashed', label='$N^{-2}$')
    # ax.loglog([RES[0], RES[-1]], 0.001*np.asarray([float(RES[0]), float(RES[-1])])**(-2), color='k', linestyle='dashed', label='$N^{-2}$')
    plt.xscale('log', base=2)
    ax.set_xlabel('Resolution')
    ax.set_ylabel('L1 norm')
    ax.legend()
    plt.savefig(os.path.join(outputdir, "bondi_viscous_convergence_"+SHORT+".png"), dpi=300)

    exit(fail)
