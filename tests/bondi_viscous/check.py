import numpy as np
import os, glob, h5py, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt


if __name__=='__main__':
	outputdir = os.getcwd()
	kharmadir = '/data/bh29-home/vdhruv2/kharma'
	RES = [int(r) for r in sys.argv[1].split(",")]
	VISCOSITY = 1
	if VISCOSITY:
			PRIMS = ['rho','u','dP']
	else:
			PRIMS = ['rho','u']
	L1_norm = np.zeros([len(RES), len(PRIMS)])

	for r, res in enumerate(RES):
			
		# load analytic result
		if VISCOSITY:
			rho_analytic, u_analytic, dP_analytic = np.loadtxt(os.path.join(kharmadir, \
			'kharma/prob/emhd/','bondi_viscous_{}_default'.format(res), 'bondi_analytic_{}.txt'.format(res)), \
			usecols=(0,1,3), unpack=True)
		else:
			rho_analytic, u_analytic, = np.loadtxt(os.path.join(kharmadir, \
			'kharma/prob/emhd/','bondi_viscous_{}_default'.format(res), 'bondi_analytic_{}.txt'.format(res)), \
			usecols=(0,1), unpack=True)
		
		# load code data
		dfile = h5py.File('emhd_2d_{}_end.h5'.format(res), 'r')
		
		rho       = np.squeeze(dfile['prims'][Ellipsis,0][()])
		u         = np.squeeze(dfile['prims'][Ellipsis,1][()])
		if VISCOSITY:
			dP_tilde   = np.squeeze(dfile['prims'][Ellipsis,9][()])
		
		t   = dfile['t'][()]
		gam = dfile['header/gam'][()]
		higher_order_terms = dfile['header/higher_order_terms'][()].decode('UTF-8')

		# compute dP
		if VISCOSITY:
			if higher_order_terms=="TRUE":
				tau      = 30.
				eta      = 0.01
				P        = (gam - 1.) * u
				Theta    = P / rho
				nu_emhd  = eta / rho
				dP       = dP_tilde * np.sqrt(nu_emhd * rho * Theta / tau)
			else:
				dP = dP_tilde
		
		# compute L1 norm
		L1_norm[r,0] = np.mean(np.fabs(rho-rho_analytic[:,None]))
		L1_norm[r,1] = np.mean(np.fabs(u-u_analytic[:,None]))
		if VISCOSITY:
			L1_norm[r,2] = np.mean(np.fabs(dP-dP_analytic[:,None])[1:-1])
			
			
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
	for n in range(len(PRIMS)):
			color = colors[tracker]
			ax.loglog(RES, L1_norm[:,n], color=color, marker='o', label=PRIMS[n])
			tracker+=1

	ax.loglog([RES[0], RES[-1]], 0.1*np.asarray([float(RES[0]), float(RES[-1])])**(-2), color='k', linestyle='dashed', label='$N^{-2}$')
	plt.xscale('log', base=2)
	ax.set_xlabel('Resolution')
	ax.set_ylabel('L1 norm')
	ax.legend()
	plt.savefig(os.path.join(outputdir, 'bondi_viscous_convergence.png'), dpi=300)
