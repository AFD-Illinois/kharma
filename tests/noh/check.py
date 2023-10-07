import numpy as np
import os, sys, h5py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pyharm

if __name__=='__main__':
    plotsdir = sys.argv[1]
    filesdir = sys.argv[2]
    resolutions = sys.argv[3].split(',')
    for r, resolution in enumerate(resolutions):
        resolutions[r] = int(resolution)
    resolutions = np.array(resolutions)
    gamma_e = float(sys.argv[4])

    l1_norm = []

    # read data
    for r, resolution in enumerate(resolutions):
        #hfp = h5py.File(os.path.join(filesdir, 'noh.out0.final.res{:d}.h5'.format(resolution)))
        hfp = pyharm.load_dump('noh.out0.final.res{:d}.phdf'.format(resolution))
        gam = hfp['gam']
        gam_e = hfp['gam_e']
        fel = hfp['electrons/fel_constant']
        rho = np.squeeze(hfp['rho'])
        uu = np.squeeze(hfp['u'])
        kel = np.squeeze(hfp['Kel_Constant'])
        startx1 = hfp['startx1']
        dx1 = hfp['dx1']
        n1 = hfp['n1']
        del hfp

        x1 = np.zeros(n1, dtype=float)
        for i in range(n1):
            x1[i] = startx1 + i*dx1

        u_e = (kel * rho**gam_e)/(gam_e - 1.)
        ratio_analytical = np.where(rho > 1.5, \
                                    fel/2. * (((gam + 1.)/(gam - 1.))**gam_e * (1. - gam/gam_e) + 1. + gam/gam_e) * ((gam**2 - 1.)/(gam_e**2 - 1.)), \
                                    0.)

        plt.figure(figsize=(6,6))
        plt.plot(x1, u_e/uu, label="Computed")
        plt.plot(x1, ratio_analytical*np.ones_like(x1), label="Analytic")
        plt.legend()
        plt.savefig("noh_results_{}.png".format(resolution))

        l1_norm.append(np.mean(abs(u_e/uu - ratio_analytical)))
    
    l1_norm = np.array(l1_norm)
    powerfit = np.polyfit(np.log(resolutions), np.log(l1_norm), 1)[0]
    print("Power fit: {} {}".format(powerfit, l1_norm))
    # These bounds were chosen heuristically
    if powerfit < -0.85 and powerfit > -1.15:
        fail = 0
    else:
        fail = 1

    # plot
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    ax.plot(resolutions, l1_norm, color='darkblue', marker='^', markersize=8, label='$\\gamma_{{e}}$={:.2f}'.format(gamma_e))
    start_val = float(resolutions[0])*l1_norm[0]
    ax.plot([resolutions[0], resolutions[-1]], start_val*np.asarray([float(resolutions[0]), float(resolutions[-1])])**(-1), color='black', linestyle='dashed', label='$N^{-1}$')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Resolution')
    plt.ylabel('L1 Norm')
    plt.legend()
    plt.savefig(os.path.join(plotsdir, 'noh_convergence_{:.2f}.png'.format(gamma_e)), dpi=200)
    plt.close()

    exit(fail)
