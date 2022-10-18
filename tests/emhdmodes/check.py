import numpy as np
import os, sys, h5py, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

from pyharm.grid import make_some_grid

if __name__=='__main__':
    outputdir = './'

    NVAR = 10
    VARS = ['rho', 'u', 'u1', 'u2', 'u3', 'B1', 'B2', 'B3', 'q', 'deltaP']
    RES = [int(r) for r in sys.argv[1].split(",")]

    # problem params
    var0 = np.zeros(NVAR)
    var0[0] = 1.
    var0[1] = 2.
    var0[5] = 0.1
    var0[6] = 0.3

    # L1 initialization
    L1 = np.zeros([len(RES), NVAR])
    fit = np.zeros([len(RES), NVAR])

    # perturbation (for 2D EMHD wave)
    dvar_cos = np.zeros(NVAR)
    dvar_cos[0] = -0.518522524082246
    dvar_cos[1] = 0.5516170736393813
    dvar_cos[2] = 0.008463122479547856
    dvar_cos[3] = -0.16175466371870734
    dvar_cos[5] = -0.05973794979640743
    dvar_cos[6] = 0.02986897489820372
    dvar_cos[8] = 0.5233486841539436
    dvar_cos[9] = 0.2909106062057657
    dvar_sin = np.zeros(NVAR)
    dvar_sin[0] = 0.1792647678001878
    dvar_sin[2] = -0.011862022608466367
    dvar_sin[3] = 0.034828080823603294
    dvar_sin[5] = 0.03351707506150924
    dvar_sin[6] = -0.016758537530754618
    dvar_sin[8] = -0.04767672501939603
    dvar_sin[9] = -0.02159452055336572

    # loop over RES
    for r in range(len(RES)):
        # load data
        dfile = h5py.File("emhd_2d_"+str(RES[r])+"_end_"+sys.argv[3]+".h5", 'r')

        dump = {}

        amp = float(dfile['header/amp'][()])
        k1  = 2*np.pi
        k2  = 4*np.pi
        real_omega  = dfile['header/omega_real'][()]
        imag_omega  = dfile['header/omega_imag'][()]
        t = dfile['t'][()]

        dump['RHO'] = dfile['prims'][Ellipsis,0][()]
        dump['U'] = dfile['prims'][Ellipsis,1][()]
        dump['U1'] = dfile['prims'][Ellipsis,2][()]
        dump['U2'] = dfile['prims'][Ellipsis,3][()]
        dump['U3'] = dfile['prims'][Ellipsis,4][()]
        dump['B1'] = dfile['prims'][Ellipsis,5][()]
        dump['B2'] = dfile['prims'][Ellipsis,6][()]
        dump['B3'] = dfile['prims'][Ellipsis,7][()]
        dump['q'] = dfile['prims'][Ellipsis,8][()]
        dump['deltaP'] = dfile['prims'][Ellipsis,9][()]

        gridp = {}
        gridp['n1'] = dfile['header/n1'][()]
        gridp['n2'] = dfile['header/n2'][()]
        gridp['n3'] = dfile['header/n3'][()]
        
        higher_order_terms = dfile['header/higher_order_terms'][()].decode('UTF-8')
        gam                = dfile['header/gam'][()]
        tau                = dfile['header/tau'][()]
        conduction_alpha   = dfile['header/conduction_alpha'][()]
        viscosity_alpha    = dfile['header/viscosity_alpha'][()]

        grid = make_some_grid('cartesian', gridp['n1'], gridp['n2'], gridp['n3'])
        cos_phi = np.cos(k1*grid['x'] + k2*grid['y'] + imag_omega*t)
        sin_phi = np.sin(k1*grid['x'] + k2*grid['y'] + imag_omega*t)

        dfile.close()

        # compute analytic result
        var_analytic  = []
        for i in range(NVAR):
            var_analytic.append(var0[i] + ((amp*cos_phi*dvar_cos[i]) + (amp*sin_phi*dvar_sin[i])) * np.exp(real_omega*t))
        var_analytic = np.asarray(var_analytic)

        # numerical result
        # TODO 3D, but will need different coeffs too
        var_numerical = np.zeros((NVAR, grid['n1'], grid['n2']), dtype=float)
        var_numerical[0,Ellipsis] = dump['RHO'] 
        var_numerical[1,Ellipsis] = dump['U'] 
        var_numerical[2,Ellipsis] = dump['U1'] 
        var_numerical[3,Ellipsis] = dump['U2'] 
        var_numerical[4,Ellipsis] = dump['U3'] 
        var_numerical[5,Ellipsis] = dump['B1'] 
        var_numerical[6,Ellipsis] = dump['B2'] 
        var_numerical[7,Ellipsis] = dump['B3'] 
        var_numerical[8,Ellipsis] = dump['q'] 
        var_numerical[9,Ellipsis] = dump['deltaP']
        
        if higher_order_terms=="TRUE":
            Theta = (gam - 1.) * dump['U'] / dump['RHO']
            cs2   = gam * (gam - 1.) * dump['U'] / (dump['RHO'] + (gam * dump['U']) )

            var_numerical[8,Ellipsis] *= np.sqrt(conduction_alpha * cs2 * dump['RHO'] * Theta**2)
            var_numerical[9,Ellipsis] *= np.sqrt(viscosity_alpha * cs2 * dump['RHO'] * Theta)

        print("\n{:d}".format(RES[r]))
        print(np.mean(np.fabs(var_numerical - var_analytic)[6,Ellipsis]))

        for n in range(NVAR):
            L1[r,n] = np.mean(np.fabs(var_numerical[n,Ellipsis] - var_analytic[n,Ellipsis]))

    # plot parameters
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
    colors = ['indigo', 'goldenrod', 'darkgreen', 'crimson', 'xkcd:blue', 'xkcd:magenta', 'green', 'xkcd:yellowgreen', 'xkcd:teal', 'xkcd:olive']

    # plot
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)

    fig.suptitle(sys.argv[2])

    # loop over prims
    tracker = 0
    for n in range(NVAR):
        if abs((dvar_cos[n] != 0) or abs(dvar_sin[n] != 0)):
            color = colors[tracker]
            ax.loglog(RES, L1[:,n], color=color, marker='o', label=VARS[n])
            tracker+=1

    ax.loglog([RES[0], RES[-1]], 100*amp*np.asarray([float(RES[0]), float(RES[-1])])**(-2), color='k', linestyle='dashed', label='$N^{-2}$')
    plt.xscale('log', base=2)
    ax.legend()
    plt.savefig(os.path.join(outputdir, "emhd_linear_mode_convergence_"+sys.argv[3]+".png"))
