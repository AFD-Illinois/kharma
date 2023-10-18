import numpy as np
import os, sys, h5py, glob
from scipy import optimize
from scipy.interpolate import splrep, splev
from scipy.integrate import odeint, solve_ivp
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import pyharm
import pyharm.io.gridfile as gridfile

# Global dictionaries to store (i) fluid dump (ii) grid (iii) analytic solution data
dump = {}
grid = {}
soln = {}

############### GEOMETRY FUNCTIONS ###############
# Compute gcov in BL from (r,th,phi) read from grid file
def gcov_bl():
    grid['gcov_bl'] = np.zeros_like(grid['gcov'])

    DD = 1 - 2./grid['r'] + grid['a']**2/grid['r']**2
    mu = 1 + grid['a']**2 * np.cos(grid['th'])**2 / grid['r']**2

    grid['gcov_bl'][Ellipsis,0,0] = -(1 - 2./(grid['r'] * mu))
    grid['gcov_bl'][Ellipsis,0,3] = -2 * grid['a'] * np.sin(grid['th'])**2 / (grid['r'] * mu)
    grid['gcov_bl'][Ellipsis,3,0] = grid['gcov_bl'][Ellipsis,0,3]
    grid['gcov_bl'][Ellipsis,1,1] = mu / DD
    grid['gcov_bl'][Ellipsis,2,2] = grid['r']**2 * mu
    grid['gcov_bl'][Ellipsis,3,3] = grid['r']**2 * np.sin(grid['th'])**2 * (1 + grid['a']**2/grid['r']**2 \
                                    + 2 * grid['a']**2 * np.sin(grid['th'])**2 / (grid['r']**3 * mu))

# Compute gcov in KS from (r,th,phi) read from grid file
def gcov_ks():
    grid['gcov_ks'] = np.zeros_like(grid['gcov'])
    sigma = grid['r']**2 + (grid['a']**2 * np.cos(grid['th'])**2)
    
    grid['gcov_ks'][Ellipsis,0,0] = -1 + 2*grid['r']/sigma
    grid['gcov_ks'][Ellipsis,0,1] = 2*grid['r']/sigma
    grid['gcov_ks'][Ellipsis,0,3] = -(2*grid['a']*grid['r']*np.sin(grid['th'])**2)/sigma
    grid['gcov_ks'][Ellipsis,1,0] = 2*grid['r']/sigma
    grid['gcov_ks'][Ellipsis,1,1] = 1 + 2*grid['r']/sigma
    grid['gcov_ks'][Ellipsis,1,3] = -grid['a']*np.sin(grid['th'])**2 * (1 + 2*grid['r']/sigma)
    grid['gcov_ks'][Ellipsis,2,2] = sigma
    grid['gcov_ks'][Ellipsis,3,0] = -(2*grid['a']*grid['r']*np.sin(grid['th'])**2)/sigma
    grid['gcov_ks'][Ellipsis,3,1] = -grid['a']*np.sin(grid['th'])**2 * (1 + 2*grid['r']/sigma)
    grid['gcov_ks'][Ellipsis,3,3] = np.sin(grid['th'])**2 * (sigma + grid['a']**2*np.sin(grid['th'])**2 * (1 + 2*grid['r']/sigma))

# Compute gcov in KS from gcon_ks
def gcon_ks():
    grid['gcon_ks'] = np.linalg.inv(grid['gcov_ks'])

# Compute transformation matrix from KS -> MKS / FMKS (for covariant indices)
def dxdX_KS_to_FMKS():
    dxdX = np.zeros((grid['n1'], grid['n2'], 4, 4), dtype=float)

    if grid['metric'] == 'mks':
        dxdX[Ellipsis,0,0] = dxdX[Ellipsis,3,3] = 1
        dxdX[Ellipsis,1,1] = np.exp(grid['x1'])
        dxdX[Ellipsis,2,2] = np.pi + (1 - grid['hslope']) * np.pi * np.cos(2 * np.pi * grid['x2'])
    
    else:
        theta_g = (np.pi * grid['x2']) + ((1 - grid['hslope'])/2) * (np.sin(2*np.pi*grid['x2']))
        theta_j = grid['D'] * (2*grid['x2'] - 1) * (1 + (((2 * grid['x2'] - 1) / grid['poly_xt'])**grid['poly_alpha']) / (1 + grid['poly_alpha'])) + np.pi/2
        derv_theta_g = np.pi + (1 - grid['hslope']) * np.pi * np.cos(2 * np.pi * grid['x2'])
        derv_theta_j = (2 * grid['poly_alpha'] * grid['D'] * (2 * grid['x2'] - 1)*((2 * grid['x2'] - 1) / grid['poly_xt'])**(grid['poly_alpha'] - 1)) / (grid['poly_xt'] * (grid['poly_alpha'] + 1)) + 2 * grid['D'] * (1 + (((2 * grid['x2'] - 1) / grid['poly_xt'])**grid['poly_alpha']) / (grid['poly_alpha'] + 1))
        dxdX[Ellipsis,0,0] = dxdX[Ellipsis,3,3] = 1
        dxdX[Ellipsis,1,1] = np.exp(grid['x1'])
        dxdX[Ellipsis,2,1] = -grid['mks_smooth'] * np.exp(-grid['mks_smooth'] * grid['Dx1'][:,np.newaxis]) * (theta_j - theta_g)
        dxdX[Ellipsis,2,2] = derv_theta_g + np.exp(-grid['mks_smooth'] * grid['Dx1'][:,np.newaxis]) * (derv_theta_j - derv_theta_g)

    return dxdX

# Compute transformation matrix from MKS / FMKS -> KS (for covariant indices)
def dxdX_FMKS_to_KS():
    return (np.linalg.inv(dxdX_KS_to_FMKS()))

# Compute quantities manually from x^mu
def bl_coords_from_x(grid_temp):
    grid_temp['r']  = np.exp(grid_temp['x1'])
    grid_temp['th'] = np.pi * grid_temp['x2'] + ((1 - grid['hslope'])/2.) * np.sin(2*np.pi*grid_temp['x2'])

def gcov_ks_from_x(grid_temp):
    bl_coords_from_x(grid_temp)

    grid_temp['gcov_ks'] = np.zeros_like(grid['gcov'])
    sigma = grid_temp['r']**2 + (grid_temp['a']**2 * np.cos(grid_temp['th'])**2)
    
    grid_temp['gcov_ks'][Ellipsis,0,0] = -1 + 2*grid_temp['r']/sigma
    grid_temp['gcov_ks'][Ellipsis,0,1] = 2*grid_temp['r']/sigma
    grid_temp['gcov_ks'][Ellipsis,0,3] = -(2*grid_temp['a']*grid_temp['r']*np.sin(grid_temp['th'])**2)/sigma
    grid_temp['gcov_ks'][Ellipsis,1,0] = 2*grid_temp['r']/sigma
    grid_temp['gcov_ks'][Ellipsis,1,1] = 1 + 2*grid_temp['r']/sigma
    grid_temp['gcov_ks'][Ellipsis,1,3] = -grid_temp['a']*np.sin(grid_temp['th'])**2 * (1 + 2*grid_temp['r']/sigma)
    grid_temp['gcov_ks'][Ellipsis,2,2] = sigma
    grid_temp['gcov_ks'][Ellipsis,3,0] = -(2*grid_temp['a']*grid_temp['r']*np.sin(grid_temp['th'])**2)/sigma
    grid_temp['gcov_ks'][Ellipsis,3,1] = -grid_temp['a']*np.sin(grid_temp['th'])**2 * (1 + 2*grid_temp['r']/sigma)
    grid_temp['gcov_ks'][Ellipsis,3,3] = np.sin(grid_temp['th'])**2 * (sigma + grid_temp['a']**2*np.sin(grid_temp['th'])**2 * (1 + 2*grid_temp['r']/sigma))

def dxdX_KS_to_MKS_from_x(grid_temp):
    dxdX = np.zeros((grid['n1'], grid['n2'], 4, 4), dtype=float)

    dxdX[Ellipsis,0,0] = dxdX[Ellipsis,3,3] = 1
    dxdX[Ellipsis,1,1] = np.exp(grid_temp['x1'])
    dxdX[Ellipsis,2,2] = np.pi + (1 - grid['hslope']) * np.pi * np.cos(2 * np.pi * grid_temp['x2'])

    return dxdX

def dxdX_MKS_to_KS_from_x(grid_temp):
    dxdX = dxdX_KS_to_MKS_from_x(grid_temp)
    return np.linalg.inv(dxdX)

def gcov_from_x(grid_temp):
    gcov_ks_from_x(grid_temp)
    dxdX = dxdX_KS_to_MKS_from_x(grid_temp)

    grid_temp['gcov'] = np.einsum('ijbn,ijmb->ijmn', dxdX, \
                        np.einsum('ijam,ijab->ijmb', dxdX, grid_temp['gcov_ks']))

    grid_temp['gcon'] = np.linalg.inv(grid_temp['gcov'])

# Compute the Christoffel symbols in MKS/MMKS (like iharm3d/pyharm)
def conn_func(sigma, alpha, beta):
    delta = 1.e-5
    conn = np.zeros((grid['n1'], grid['n2'], 4, 4, 4), dtype=float)
    tmp  = np.zeros_like(conn)

    x = np.zeros((grid['n1'], grid['n2'], 4), dtype=float)
    x[Ellipsis,1] = grid['x1']
    x[Ellipsis,2] = grid['x2']
    x[Ellipsis,3] = grid['x3']

    grid_h = {}; grid_h['a'] = grid['a']
    grid_l = {}; grid_l['a'] = grid['a']

    for mu in range(4):
        xh = np.copy(x)
        xl = np.copy(x)
        xh[Ellipsis,mu] += delta
        xl[Ellipsis,mu] -= delta

        grid_h['x1'] = xh[Ellipsis,1]
        grid_h['x2'] = xh[Ellipsis,2]
        grid_l['x1'] = xl[Ellipsis,1]
        grid_l['x2'] = xl[Ellipsis,2]

        gcov_from_x(grid_h)
        gcov_from_x(grid_l)

        for lam in range(4):
            for nu in range(4):
                conn[Ellipsis,lam,nu,mu] = (grid_h['gcov'][Ellipsis,lam,nu] - grid_l['gcov'][Ellipsis,lam,nu]) \
                                            / (xh[Ellipsis,mu] - xl[Ellipsis,mu])

    for lam in range(4):
        for nu in range(4):
            for mu in range(4):
                tmp[Ellipsis,lam,nu,mu] = 0.5 * (conn[Ellipsis,nu,lam,mu] + conn[Ellipsis,mu,lam,nu] \
                - conn[Ellipsis,mu,nu,lam])

    for lam in range(4):
        for nu in range(4):
            for mu in range(4):
                conn[Ellipsis,lam,nu,mu] = 0
                for kap in range(4):
                    conn[Ellipsis,lam,nu,mu] += grid['gcon'][Ellipsis,lam,kap] * tmp[Ellipsis,kap,nu,mu]

    return conn[Ellipsis,sigma,alpha,beta]



############### READ DATA ###############
# Read dump and/or grid file
def load_data(dumpsdir, dumpno, read_grid=False):
    dfile = pyharm.load_dump(dumpsdir+'/emhd_2d_8_end_emhd2d_weno.phdf')
    dump['rc']    = dfile['rs']
    dump['mdot']  = dfile['mdot']
    dump['gam']   = dfile['gam']
    dump['rEH']   = dfile['r_eh']

    if read_grid:
        gridfile.write_grid(dfile.grid, 'grid.h5')
        gfile  = h5py.File(os.path.join(dumpsdir, 'grid.h5'), 'r')
        grid['r']   = np.squeeze(gfile['r'])
        grid['th']  = np.squeeze(gfile['th'])
        grid['phi'] = np.squeeze(gfile['phi'])

        grid['rEH_ind'] = np.argmin(np.fabs(grid['r'][:,0]-dump['rEH']) > 0.)
        grid['n1']  = dfile['n1']
        grid['n2']  = dfile['n2']
        grid['n3']  = dfile['n3']
        grid['dx1'] = dfile['dx1']
        grid['dx2'] = dfile['dx2']

        grid['x1'] = np.squeeze(gfile['X1'])
        grid['x2'] = np.squeeze(gfile['X2'])
        grid['x3'] = np.squeeze(gfile['X3'])

        grid['metric'] = dfile['coordinates'].lower()
        grid['gcov']   = np.squeeze(gfile['gcov'])
        grid['gcon']   = np.squeeze(gfile['gcon'])
        grid['gdet']   = np.squeeze(gfile['gdet'])
        grid['lapse']  = np.squeeze(gfile['lapse'])

        if grid['metric']=='mks' or grid['metric']=='mmks':
            grid['a'] = dfile['a']
            grid['rEH'] = dfile['r_eh']
            grid['hslope'] = dfile['hslope']

        if grid['metric']=='MMKS':
            grid['mks_smooth'] = dfile['mks_smooth']
            grid['poly_alpha'] = dfile['poly_alpha']
            grid['poly_xt'] = dfile['poly_xt']
            grid['D'] = (np.pi*grid['poly_xt']**grid['poly_alpha'])/(2*grid['poly_xt']**grid['poly_alpha']+(2/(1+grid['poly_alpha'])))

        gfile.close()

    del dfile



############### COMPUTE ANALYTIC IDEAL BONDI SOLUTION ###############
# Nonlinear expression to solve for T
def T_func(T, r, C3, C4, N):
    return (1 + (1 + N/2)*T)**2 * (1 - 2./r + (C4**2/(r**4 * T**N))) - C3

# Obtain primitives for Bondi problem
def get_prim():
    N    = 2./ (dump['gam'] - 1)
    rc   = dump['rc']
    mdot = dump['mdot']
    vc   = np.sqrt(1. / (2 * rc))
    csc  = np.sqrt(vc**2 / (1 - 3*vc**2))
    Tc   = 2*N*csc**2 / ((N + 2)*(2 - N*csc**2))
    C4   = Tc**(N/2)*vc*rc**2
    C3   = (1 + (1 + N/2)*Tc)**2 * (1 - 2./rc + vc**2)

    # Root find T
    T = np.zeros_like(grid['r'][:,0])
    for index, r in enumerate(grid['r'][:,0]):
        T0       = Tc
        sol      = optimize.root(T_func, [T0], args=(r, C3, C4, N))
        T[index] = sol.x[0]
        if (sol.success!=True):
            print("Not converged at r = {:.2f}", r)

    # Compute remaining fluid variables
    soln['T'] = T
    soln['v'] = -C4 / (T**(N/2) * grid['r'][:,0]**2)
    soln['K'] = (4*np.pi*C4 / mdot) ** (2./N)

    soln['rho'] = soln['K']**(-N/2) * T**(N/2)
    soln['u']   = (N/2) * soln['K']**(-N/2) * T**(N/2 + 1)

    soln['mdot'] = mdot
    soln['N']    = N
    soln['rc']   = rc

# Compute four vectors
def compute_ub():

    # We have u^r in BL. We need to convert this to ucon in MKS
    # First compute u^t in BL
    ucon_bl = np.zeros((grid['n1'], grid['n2'], 4), dtype=float)
    AA = grid['gcov_bl'][Ellipsis,0,0]
    BB = 2. * grid['gcov_bl'][Ellipsis,0,1]*soln['v'][:,None]
    CC = 1. + grid['gcov_bl'][Ellipsis,1,1]*soln['v'][:,None]**2
    
    discr = BB*BB - 4.*AA*CC
    ucon_bl[Ellipsis,0] = (-BB - np.sqrt(discr)) / (2.*AA)
    ucon_bl[Ellipsis,1] = soln['v'][:,None]

    # Convert ucon(Bl) to ucon(KS)
    dxdX = np.zeros((grid['n1'], grid['n2'], 4, 4), dtype=float)
    dxdX[Ellipsis,0,0] = dxdX[Ellipsis,1,1] = dxdX[Ellipsis,2,2] = dxdX[Ellipsis,3,3] = 1.
    dxdX[Ellipsis,0,1] = 2*grid['r'] / (grid['r']**2 - 2.*grid['r'] + grid['a']**2)
    dxdX[Ellipsis,3,1] = grid['a']/(grid['r']**2 - 2.*grid['r'] + grid['a']**2)

    ucon_ks = np.zeros((grid['n1'], grid['n2'], 4), dtype=float)
    for mu in range(4):
        for nu in range(4):
            ucon_ks[Ellipsis,mu] += dxdX[Ellipsis,mu,nu] * ucon_bl[Ellipsis,nu]

    # Convert ucon(KS) to ucon(MKS/FMKS)
    ucon_mks = np.zeros((grid['n1'], grid['n2'], 4), dtype=float)
    dxdX = dxdX_FMKS_to_KS()
    for mu in range(4):
        for nu in range(4):
            ucon_mks[Ellipsis,mu] += dxdX[Ellipsis,mu,nu] * ucon_ks[Ellipsis,nu]

    ucov_mks = np.einsum('ijmn,ijn->ijm', grid['gcov'], ucon_mks)

    # Compute velocity primitives
    utilde = np.zeros((grid['n1'], grid['n2'], 3), dtype=float)

    alpha = 1./np.sqrt(-grid['gcon'][Ellipsis,0,0])
    beta  = np.zeros((grid['n1'], grid['n2'], 3), dtype=float)
    beta[Ellipsis,0] = alpha * alpha * grid['gcon'][Ellipsis,0,1]
    beta[Ellipsis,1] = alpha * alpha * grid['gcon'][Ellipsis,0,2]
    beta[Ellipsis,2] = alpha * alpha * grid['gcon'][Ellipsis,0,3]
    gamma = ucon_mks[Ellipsis,0] * alpha

    utilde[Ellipsis,0] = ucon_mks[Ellipsis,1] + beta[Ellipsis,0]*gamma/alpha
    utilde[Ellipsis,1] = ucon_mks[Ellipsis,2] + beta[Ellipsis,1]*gamma/alpha
    utilde[Ellipsis,2] = ucon_mks[Ellipsis,3] + beta[Ellipsis,2]*gamma/alpha

    # compute magnetic 4-vector
    B = np.zeros((grid['n1'], grid['n2'], 3), dtype=float)
    # radial magnetic field (B1 = 1/r^3)
    B[Ellipsis,0] = 1. / grid['r']**3

    gti    = grid['gcon'][Ellipsis,0,1:4]
    gij    = grid['gcov'][Ellipsis,1:4,1:4]
    beta_i = np.einsum('ijs,ij->ijs', gti, grid['lapse']**2)
    qsq    = np.einsum('ijy,ijy->ij', np.einsum('ijxy,ijx->ijy', gij, utilde), utilde)
    gamma  = np.sqrt(1 + qsq)
    ui     = utilde - np.einsum('ijs,ij->ijs', beta_i, gamma/grid['lapse'])
    ut     = gamma/grid['lapse']

    bt = np.einsum('ijm,ijm->ij', np.einsum('ijsm,ijs->ijm', grid['gcov'][Ellipsis,1:4,:], B), ucon_mks)
    bi = (B + np.einsum('ijs,ij->ijs', ucon_mks[Ellipsis,1:4], bt)) / ucon_mks[Ellipsis,0,None]
    bcon_mks = np.append(bt[Ellipsis,None], bi, axis=2)
    bcov_mks = np.einsum('ijmn,ijn->ijm', grid['gcov'], bcon_mks)

    soln['ucon'] = ucon_mks[:,0,:]
    soln['ucov'] = ucov_mks[:,0,:]
    soln['bcon'] = bcon_mks[:,0,:]
    soln['bcov'] = bcov_mks[:,0,:]
    soln['bsq']  = np.einsum('im,im->i', soln['bcon'], soln['bcov'])



############### ADDITIONAL FUNCTIONS FOR VISCOUS BONDI FLOW ###############
# Compute Braginskii pressure anisotropy value
def compute_dP0():
    grid['dx'] = [grid['dx1'],grid['dx2']]

    soln['tau'] = 30.
    soln['eta'] = 0.01
    nu_emhd     = soln['eta'] / soln['rho']
    dP0         = np.zeros(grid['n1'], dtype=float)

    # Compute derivatives of 4-velocity
    ducovDx1 = np.zeros((grid['n1'], 4), dtype=float) # Represents d_x1(u_\mu)
    delta = 1.e-5
    x1    = grid['x1'][:,0]
    x1h   = x1 + delta
    x1l   = x1 - delta

    ucovt_splrep = splrep(x1, soln['ucov'][:,0])
    ucovr_splrep = splrep(x1, soln['ucov'][:,1])
    ucovt_h = splev(x1h, ucovt_splrep) 
    ucovt_l = splev(x1l, ucovt_splrep) 
    ucovr_h = splev(x1h, ucovr_splrep) 
    ucovr_l = splev(x1l, ucovr_splrep)

    ducovDx1[:,0] = (ucovt_h - ucovt_l) / (x1h - x1l)
    ducovDx1[:,1] = (ucovr_h - ucovr_l) / (x1h - x1l)

    for mu in range(4):
        for nu in range(4):
            if mu == 1:
                dP0 += 3*soln['rho']*nu_emhd * (soln['bcon'][:,mu]*soln['bcon'][:,nu] / soln['bsq']) \
                        * ducovDx1[:,nu]
                
            gamma_term_1 = np.zeros((grid['n1'], grid['n2']), dtype=float)
            for sigma in range(4):
                gamma_term_1 += (3*soln['rho']*nu_emhd * (soln['bcon'][:,mu]*soln['bcon'][:,nu] / soln['bsq']))[:,None] \
                                * (-conn_func(sigma, mu, nu) * soln['ucov'][:,None,sigma])

            dP0 += np.mean(gamma_term_1, axis=1)

        derv_term_2 = np.zeros((grid['n1'], grid['n2']), dtype=float)
        if mu == 1:
            for sigma in range(4):
                derv_term_2 += (-soln['rho']*nu_emhd * ducovDx1[:,sigma])[:,None] \
                                * grid['gcon'][Ellipsis,mu,sigma]

        dP0 += np.mean(derv_term_2, axis=1)

        gamma_term_2 = np.zeros((grid['n1'], grid['n2']), dtype=float)
        for sigma in range(4):
            for delta in range(4):
                    gamma_term_2 += (soln['rho']*nu_emhd)[:,None] * (conn_func(sigma, mu, delta) * grid['gcon'][Ellipsis,mu,delta] * soln['ucov'][:,None,sigma])

        dP0 += np.mean(gamma_term_2, axis=1)

    # r_start = 3.0
    # r_start_ind = np.argmin(np.fabs(grid['r'][:,0] - r_start))
    # plt.semilogx(grid['r'][r_start_ind:,0], dP0[r_start_ind:])
    # plt.savefig('dP0_analytic.png')
    # plt.close()
    
    return dP0

# Compute the coefficient of the second term on the RHS of the evolution equation of dP
def compute_rhs_second_term():
    nu_emhd = soln['eta'] / soln['rho']
    P = soln['u'] * (dump['gam'] - 1.)

    # compute derivative
    delta = 1.e-5
    x1    = grid['x1'][:,0]
    x1h   = x1 + delta
    x1l   = x1 - delta
    expr  = np.log(soln['tau'] / (soln['rho'] * nu_emhd * P))
    expr_splrep = splrep(x1, expr)
    expr_h = splev(x1h, expr_splrep)
    expr_l = splev(x1l, expr_splrep)

    coeff  = 0.5 * (expr_h - expr_l) / (x1h - x1l)

    return coeff

# Return derivative d(dP)/dx1. Refer Equation (36) in grim paper
def ddP_dX1(dP, x1, ur_splrep, dP0_splrep, coeff_splrep):
    tau   = soln['tau']
    ur    = splev(x1, ur_splrep)
    dP0   = splev(x1, dP0_splrep)
    coeff = splev(x1, coeff_splrep)

    derivative = -((dP - dP0) / (tau * ur)) - (dP * coeff)
    return derivative


############### MAIN IS MAIN ###############
if __name__=='__main__':
    dumpsdir = '.'


    load_data(dumpsdir, 0, True)
    get_prim()
    gcov_bl()
    gcov_ks()
    gcon_ks()
    compute_ub()

    dP0   = compute_dP0()
    coeff = compute_rhs_second_term()

    x1 = grid['x1'][:,0]
    ur_splrep    = splrep(x1, soln['ucon'][:,1])
    dP0_splrep   = splrep(x1, dP0)
    coeff_splrep = splrep(x1, coeff)

    solution = odeint(ddP_dX1, 0., x1[::-1], args=(ur_splrep, dP0_splrep, coeff_splrep))
    np.savetxt('bondi_analytic_{}.txt'.format(grid['n1']), np.asarray([soln['rho'], soln['u'], soln['v'], solution[::-1,0]]).T)
    
    r_start = 3.0
    r_start_ind = np.argmin(np.fabs(grid['r'][:,0] - r_start))
    plt.plot(grid['r'][r_start_ind:,0], solution[::-1,0][r_start_ind:], label='dP ODE check')
    plt.plot(grid['r'][r_start_ind:,0], dP0[r_start_ind:], label='dP0 ODE check')
    plt.plot(grid['r'][r_start_ind:,0], soln['ucon'][:,1][r_start_ind:], label='ur')
    #plt.plot(grid['r'][r_start_ind:,0], coeff[r_start_ind:], label='coeff')
    plt.legend()
    plt.savefig('dP_soln.png')
    plt.close()
    