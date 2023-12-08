import numpy as np
import os, glob, h5py, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt

import pyharm


if __name__=='__main__':
    outputdir = './'
    kharmadir = '../../'

    NVAR = 3
    VARS  = ['rho', 'u', 'q']
    NG    = 4
    RES   = [int(r) for r in sys.argv[1].split(",")]
    LONG  = sys.argv[2]
    SHORT = sys.argv[3]

    L1  = np.zeros([len(RES), NVAR])
    fit = np.zeros([len(RES), NVAR])

    for r, res in enumerate(RES):
            
        # load analytic result
        folder = os.path.join(os.curdir,'conducting_atmosphere_{}_default'.format(res))
        rho_analytic = np.loadtxt(os.path.join(folder, 'atmosphere_soln_rho.txt'))[NG:-NG]
        uu_analytic  = np.loadtxt(os.path.join(folder, 'atmosphere_soln_u.txt'))[NG:-NG]
        q_analytic   = np.loadtxt(os.path.join(folder, 'atmosphere_soln_phi.txt'))[NG:-NG]

        # load code data
        dfile = h5py.File('emhd_2d_{}_end_emhd2d_weno.h5'.format(res), 'r')
        
        rho       = np.squeeze(dfile['prims'][Ellipsis,0][()])
        uu        = np.squeeze(dfile['prims'][Ellipsis,1][()])
        q_tilde   = np.squeeze(dfile['prims'][Ellipsis,8][()])
        
        t   = dfile['t'][()]
        gam = dfile['header/gam'][()]

        # compute q
        if dfile['header/higher_order_terms']:
            print("Res: "+str(res)+"; higher order terms enabled")
            tau      = 10.
            kappa    = 0.1
            P        = (gam - 1.) * uu
            Theta    = P / rho
            chi_emhd = kappa / rho
            q        = q_tilde * np.sqrt(chi_emhd * rho * Theta**2 / tau)
        else:
            q = q_tilde

        fig = plt.figure(figsize=(8,8))
        plt.plot(np.mean(q, axis=-1))
        plt.plot(q_analytic)
        plt.savefig("compare_{}.png".format(res))

        fig = plt.figure(figsize=(8,8))
        plt.plot(np.mean(q - q_analytic[:,None], axis=-1))
        plt.savefig("diff_{}.png".format(res))

        # fig, ax = plt.subplots(1,1,figsize=(8,8))
        # pplt.plot_xy(q - q_analytic[:,None], axis=-1))
        # plt.savefig("diff_{}.png".format(res))

        # compute L1 norm
        # compute L1 norm
        L1[r,0] = np.mean(np.fabs(rho - rho_analytic[:,None]))
        L1[r,1] = np.mean(np.fabs(uu  - uu_analytic[:,None]))
        L1[r,2] = np.mean(np.fabs(q   - q_analytic[:,None])[1:-1])

    # MEASURE CONVERGENCE
    L1 = np.array(L1)
    powerfits = [0.,]*NVAR
    fail_flag = 0
    for k in range(NVAR):
        powerfits[k] = np.polyfit(np.log(RES), np.log(L1[:,k]), 1)[0]
        print("Power fit {}: {} {}".format(VARS[k], powerfits[k], L1[:,k]))
        if powerfits[k] > -1.6 or powerfits[k] < -2.2:
            fail_flag = 1
            
            
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
    ax.loglog([RES[0], RES[-1]], 0.001*np.asarray([float(RES[0]), float(RES[-1])])**(-2), color='k', linestyle='dashed', label='$N^{-2}$')
    plt.xscale('log', base=2)
    ax.set_xlabel('Resolution')
    ax.set_ylabel('L1 norm')
    ax.legend()
    plt.savefig(os.path.join(outputdir, "conducting_atmosphere_convergence_"+SHORT+".png"), dpi=300)

    exit(fail_flag)
