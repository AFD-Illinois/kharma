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

	NVAR  = 3
	VARS  = ['rho', 'u', 'dP']
	RES   = [int(r) for r in sys.argv[1].split(",")]
	LONG  = sys.argv[2]
	SHORT = sys.argv[3]
	
	L1  = np.zeros([len(RES), NVAR])
	fit = np.zeros([len(RES), NVAR])

	for r, res in enumerate(RES):
			
		# load analytic result
		rho_analytic, uu_analytic, dP_analytic = np.loadtxt(os.path.join(kharmadir, \
		'kharma/prob/emhd/','bondi_viscous_{}_default'.format(res), 'bondi_analytic_{}.txt'.format(res)), \
		usecols=(0,1,3), unpack=True)
		
		# load code data
		dump = pyharm.load_dump("emhd_2d_{}_end_emhd2d_weno.phdf".format(res))
		
		params    = dump.params
		rho       = np.squeeze(dump['RHO'])
		uu        = np.squeeze(dump['UU'])
		dP_tilde  = np.squeeze(dump['prims'][8,Ellipsis])

		t   = dump['t']
		gam = params['gam']
		tau = params['tau']
		eta = params['eta']
		higher_order_terms = params['higher_order_terms']		

    # compute dP
		if higher_order_terms=="true":
			print("Res: "+str(res)+"; higher order terms enabled")
			P        = (gam - 1.) * uu
			Theta    = P / rho
			nu_emhd  = eta / rho
			dP       = dP_tilde * np.sqrt(nu_emhd * rho * Theta / tau)
		else:
			dP = dP_tilde
		
		# compute L1 norm
		L1[r,0] = np.mean(np.fabs(rho - rho_analytic[:,None]))
		L1[r,1] = np.mean(np.fabs(uu  - uu_analytic[:,None]))
		L1[r,2] = np.mean(np.fabs(dP  - dP_analytic[:,None])[1:-1])

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
