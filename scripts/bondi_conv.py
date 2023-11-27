import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprocessing as mp
import pyharm.coordinates as coords
import pyharm.fluid_dump as fluid_dump
import sys
from scipy import optimize
from scipy.interpolate import splrep, splev
from scipy.integrate import odeint, solve_ivp
sys.path.append('/home/samason4/kharma_cooling/pyharm')

# python3 ./scripts/bondi_conv.py

dump = {}
grid = {}
soln = {}


def gcov_bl(r):
    grid['gcov_bl'] = np.zeros_like(grid['gcov'][:,:,0,0])

    DD = 1 - 2./r + grid['a']**2/r**2
    mu = 1 + grid['a']**2 * 1 / r**2 #np.cos(th)**2 / r**2

    grid['gcov_bl'][0,0] = -(1 - 2./(r * mu))
    grid['gcov_bl'][0,3] = 0 #-2 * grid['a'] * np.sin(th)**2 / (r * mu)
    grid['gcov_bl'][3,0] = grid['gcov_bl'][0,3]
    grid['gcov_bl'][1,1] = mu / DD
    grid['gcov_bl'][2,2] = r**2 * mu
    grid['gcov_bl'][3,3] = 0 #r**2 * np.sin(th)**2 * (1 + grid['a']**2/r**2 \
                                    #+ 2 * grid['a']**2 * np.sin(th)**2 / (r**3 * mu))

def gcov_ks(r):
    grid['gcov_ks'] = np.zeros_like(grid['gcov'][:,:,0,0])
    sigma = r**2 + (grid['a']**2) # * np.cos(th)**2)
    
    grid['gcov_ks'][0,0] = -1 + 2*r/sigma
    grid['gcov_ks'][0,1] = 2*r/sigma
    grid['gcov_ks'][0,3] = 0# -(2*grid['a']*r*np.sin(th)**2)/sigma
    grid['gcov_ks'][1,0] = 2*r/sigma
    grid['gcov_ks'][1,1] = 1 + 2*r/sigma
    grid['gcov_ks'][1,3] = 0#-grid['a']*np.sin(th)**2 * (1 + 2*r/sigma)
    grid['gcov_ks'][2,2] = sigma
    grid['gcov_ks'][3,0] = 0#-(2*grid['a']*r*np.sin(th)**2)/sigma
    grid['gcov_ks'][3,1] = 0#-grid['a']*np.sin(th)**2 * (1 + 2*r/sigma)
    grid['gcov_ks'][3,3] = 0#np.sin(th)**2 * (sigma + grid['a']**2*np.sin(th)**2 * (1 + 2*r/sigma))

def gcon_ks():
    grid['gcon_ks'] = np.linalg.inv(np.transpose(grid['gcov_ks']))

def dxdX_KS_to_FMKS():
    dxdX = np.zeros((4, 4, grid['n2'], grid['n1']), dtype=float)

    if grid['metric'] == 'mks':
        dxdX[0,0,Ellipsis] = dxdX[3,3,Ellipsis] = 1
        dxdX[1,1,Ellipsis] = np.exp(grid['x1'])
        dxdX[2,2,Ellipsis] = np.pi + (1 - grid['hslope']) * np.pi * np.cos(2 * np.pi * grid['x2'])
    
    else:
        theta_g = (np.pi * grid['x2']) + ((1 - grid['hslope'])/2) * (np.sin(2*np.pi*grid['x2']))
        theta_j = grid['D'] * (2*grid['x2'] - 1) * (1 + (((2 * grid['x2'] - 1) / grid['poly_xt'])**grid['poly_alpha']) / (1 + grid['poly_alpha'])) + np.pi/2
        derv_theta_g = np.pi + (1 - grid['hslope']) * np.pi * np.cos(2 * np.pi * grid['x2'])
        derv_theta_j = (2 * grid['poly_alpha'] * grid['D'] * (2 * grid['x2'] - 1)*((2 * grid['x2'] - 1) / grid['poly_xt'])**(grid['poly_alpha'] - 1)) / (grid['poly_xt'] * (grid['poly_alpha'] + 1)) + 2 * grid['D'] * (1 + (((2 * grid['x2'] - 1) / grid['poly_xt'])**grid['poly_alpha']) / (grid['poly_alpha'] + 1))
        dxdX[0,0,Ellipsis] = dxdX[3,3,Ellipsis] = 1
        dxdX[1,1,Ellipsis] = np.exp(grid['x1'])
        dxdX[2,1,Ellipsis] = -grid['mks_smooth'] * np.exp(-grid['mks_smooth'] * grid['Dx1'][:,np.newaxis]) * (theta_j - theta_g)#this 2,1 should be 1,2 but I switched it to 2,1 so that I could take the transpose in dxdX_FMKS_to_KS()
        dxdX[2,2,Ellipsis] = derv_theta_g + np.exp(-grid['mks_smooth'] * grid['Dx1'][:,np.newaxis]) * (derv_theta_j - derv_theta_g)

    return dxdX

def dxdX_FMKS_to_KS():
    return (np.transpose(np.linalg.inv(np.transpose(dxdX_KS_to_FMKS()))))#im not terribly confident on the double transpose

def bl_coords_from_x(grid_temp):
    grid_temp['r']  = np.exp(grid_temp['x1'])
    grid_temp['th'] = np.pi * grid_temp['x2'] + ((1 - grid['hslope'])/2.) * np.sin(2*np.pi*grid_temp['x2'])

def gcov_ks_from_x(grid_temp):
    bl_coords_from_x(grid_temp)

    grid_temp['gcov_ks'] = np.zeros_like(grid['gcov'])
    sigma = grid_temp['r']**2 + (grid_temp['a']**2 * np.cos(grid_temp['th'])**2)
    
    grid_temp['gcov_ks'][0,0,Ellipsis] = -1 + 2*grid_temp['r']/sigma
    grid_temp['gcov_ks'][0,1,Ellipsis] = 2*grid_temp['r']/sigma
    grid_temp['gcov_ks'][0,3,Ellipsis] = -(2*grid_temp['a']*grid_temp['r']*np.sin(grid_temp['th'])**2)/sigma
    grid_temp['gcov_ks'][1,0,Ellipsis] = 2*grid_temp['r']/sigma
    grid_temp['gcov_ks'][1,1,Ellipsis] = 1 + 2*grid_temp['r']/sigma
    grid_temp['gcov_ks'][1,3,Ellipsis] = -grid_temp['a']*np.sin(grid_temp['th'])**2 * (1 + 2*grid_temp['r']/sigma)
    grid_temp['gcov_ks'][2,2,Ellipsis] = sigma
    grid_temp['gcov_ks'][3,0,Ellipsis] = -(2*grid_temp['a']*grid_temp['r']*np.sin(grid_temp['th'])**2)/sigma
    grid_temp['gcov_ks'][3,1,Ellipsis] = -grid_temp['a']*np.sin(grid_temp['th'])**2 * (1 + 2*grid_temp['r']/sigma)
    grid_temp['gcov_ks'][3,3,Ellipsis] = np.sin(grid_temp['th'])**2 * (sigma + grid_temp['a']**2*np.sin(grid_temp['th'])**2 * (1 + 2*grid_temp['r']/sigma))

def dxdX_KS_to_MKS_from_x(grid_temp):
    dxdX = np.zeros((4, 4, grid['n2'], grid['n1']), dtype=float)

    dxdX[0,0,Ellipsis] = dxdX[3,3,Ellipsis] = 1
    dxdX[1,1,Ellipsis] = np.exp(grid_temp['x1'])
    dxdX[2,2,Ellipsis] = np.pi + (1 - grid['hslope']) * np.pi * np.cos(2 * np.pi * grid_temp['x2'])

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

def conn_func(sigma, alpha, beta):
    delta = 1.e-5
    conn = np.zeros((4, 4, 4, grid['n2'], grid['n1']), dtype=float)
    tmp  = np.zeros_like(conn)

    x = np.zeros((4, grid['n2'], grid['n1']), dtype=float)
    x[1,Ellipsis] = grid['x1']
    x[2,Ellipsis] = grid['x2']
    x[3,Ellipsis] = grid['x3']

    grid_h = {}; grid_h['a'] = grid['a']
    grid_l = {}; grid_l['a'] = grid['a']

    for mu in range(4):
        xh = np.copy(x)
        xl = np.copy(x)
        xh[mu,Ellipsis] += delta
        xl[mu,Ellipsis] -= delta

        grid_h['x1'] = xh[1,Ellipsis]
        grid_h['x2'] = xh[2,Ellipsis]
        grid_l['x1'] = xl[1,Ellipsis]
        grid_l['x2'] = xl[2,Ellipsis]

        gcov_from_x(grid_h)
        gcov_from_x(grid_l)

        for lam in range(4):
            for nu in range(4):
                conn[mu,nu,lam,Ellipsis] = (grid_h['gcov'][nu,lam,Ellipsis] - grid_l['gcov'][nu,lam,Ellipsis]) \
                                            / (xh[mu,Ellipsis] - xl[mu,Ellipsis])

    for lam in range(4):
        for nu in range(4):
            for mu in range(4):
                tmp[mu,nu,lam,Ellipsis] = 0.5 * (conn[mu,lam,nu,Ellipsis] + conn[nu,lam,mu,Ellipsis] \
                - conn[lam,nu,mu,Ellipsis])

    for lam in range(4):
        for nu in range(4):
            for mu in range(4):
                conn[mu,nu,lam,Ellipsis] = 0
                for kap in range(4):
                    conn[mu,nu,lam,Ellipsis] += grid['gcon'][kap,lam,Ellipsis] * tmp[mu,nu,kap,Ellipsis]

    return conn[beta,alpha,sigma,Ellipsis]

def load_data(read_grid=False):
    dfile = fluid_dump.load_dump("./bondi.out0.{0:05}.phdf".format(0))
    dump['rc']    = dfile['rs']
    dump['mdot']  = dfile['mdot']
    dump['gam']   = dfile['gam']
    dump['rEH']   = dfile['r_eh'][()]

    if read_grid:
        gfile  = fluid_dump.load_dump("./bondi.out0.{0:05}.phdf".format(0))
        grid['gcov']   = np.squeeze(gfile['gcov'][()])
        grid['gcon']   = np.squeeze(gfile['gcon'][()])
        grid['gdet']   = np.squeeze(gfile['gdet'][()])
        grid['lapse']  = np.squeeze(gfile['lapse'][()])
        grid['r']   = np.squeeze(gfile['r'][()])
        grid['th']  = np.squeeze(gfile['th'][()])
        grid['phi'] = np.squeeze(gfile['phi'][()])

        grid['rEH_ind'] = np.argmin(np.fabs(grid['r'][:,0]-dump['rEH']) > 0.)
        grid['n1']  = 128
        grid['n2']  = 128
        grid['n3']  = 1
        grid['dx1'] = dfile['dx1']
        grid['dx2'] = dfile['dx2']

        grid['x1'] = np.squeeze(gfile['X1'][()])
        grid['x2'] = np.squeeze(gfile['X2'][()])
        grid['x3'] = np.squeeze(gfile['X3'][()])

        grid['metric'] = dfile['transform']#dfile['metric'][()].decode('utf-8').lower()

        if grid['metric']=='mks' or grid['metric']=='mmks':
            grid['a'] = dfile['a']
            grid['rEH'] = dfile['r_eh']
            grid['hslope'] = dfile['hslope']

        if grid['metric']=='MMKS':
            grid['mks_smooth'] = dfile['header/geom/mmks/mks_smooth'][()]
            grid['poly_alpha'] = dfile['header/geom/mmks/poly_alpha'][()]
            grid['poly_xt'] = dfile['header/geom/mmks/poly_xt'][()]
            grid['D'] = (np.pi*grid['poly_xt']**grid['poly_alpha'])/(2*grid['poly_xt']**grid['poly_alpha']+(2/(1+grid['poly_alpha'])))

def T_func(T, r, C3, C4, N):
    return (1 + (1 + N/2)*T)**2 * (1 - 2./r + (C4**2/(r**4 * T**N))) - C3

def get_prim(x1):
    N    = 2./ (dump['gam'] - 1)
    rc   = dump['rc']
    mdot = dump['mdot']
    vc   = np.sqrt(1. / (2 * rc))
    csc  = np.sqrt(vc**2 / (1 - 3*vc**2))
    Tc   = 2*N*csc**2 / ((N + 2)*(2 - N*csc**2))
    C4   = Tc**(N/2)*vc*rc**2
    C3   = (1 + (1 + N/2)*Tc)**2 * (1 - 2./rc + vc**2)

    # Root find T
    T = 0.0
    T0       = Tc
    sol      = optimize.root(T_func, [T0], args=(x1, C3, C4, N))
    T = sol.x[0]
    if (sol.success!=True):
        print("Not converged at r = {:.2f}", r)

    # Compute remaining fluid variables
    soln['T'] = T
    soln['v'] = -C4 / (T**(N/2) * x1**2)
    soln['K'] = (4*np.pi*C4 / mdot) ** (2./N)

    soln['rho'] = soln['K']**(-N/2) * T**(N/2)
    soln['u']   = (N/2) * soln['K']**(-N/2) * T**(N/2 + 1)

    soln['mdot'] = mdot
    soln['N']    = N
    soln['rc']   = rc

def compute_ub(r, th):

    # We have u^r in BL. We need to convert this to ucon in MKS
    # First compute u^t in BL
    ucon_bl = np.zeros((4), dtype=float)
    AA = grid['gcov_bl'][0,0]
    BB = 2. * grid['gcov_bl'][0,1]*soln['v']
    CC = 1. + grid['gcov_bl'][1,1]*soln['v']**2
    
    discr = BB*BB - 4.*AA*CC
    ucon_bl[0] = (-BB - np.sqrt(discr)) / (2.*AA)
    ucon_bl[1] = soln['v']

    # Convert ucon(Bl) to ucon(KS)
    dxdX = np.zeros((4, 4), dtype=float)
    dxdX[0,0] = dxdX[1,1] = dxdX[2,2] = dxdX[3,3] = 1.
    dxdX[1,0] = 2*r / (r**2 - 2.*r + grid['a']**2)#flipped 1 and zero
    dxdX[1,3] = grid['a']/(r**2 - 2.*r + grid['a']**2)#fliped 1 and 3

    ucon_ks = np.zeros((4), dtype=float)
    for mu in range(4):
        for nu in range(4):
            ucon_ks[mu] += dxdX[nu,mu] * ucon_bl[nu]

    soln['ucon_ks'] = ucon_ks

    """# Convert ucon(KS) to ucon(MKS/FMKS)
    ucon_mks = np.zeros((4, grid['n2'], grid['n1']), dtype=float)
    dxdX = dxdX_FMKS_to_KS()

    for mu in range(4):
        for nu in range(4):
            ucon_mks[mu,Ellipsis] += dxdX[nu,mu,Ellipsis] * ucon_ks[nu,Ellipsis]

    ucov_mks = np.einsum('nmji,nji->mji', grid['gcov'], ucon_mks)"""

    # Compute velocity primitives
    velocity = np.zeros((3), dtype=float)

    alpha = 1./np.sqrt(-grid['gcon_ks'][0,0])
    beta  = np.zeros((3), dtype=float)
    beta[0] = alpha * alpha * grid['gcon_ks'][1,0]
    beta[1] = alpha * alpha * grid['gcon_ks'][2,0]
    beta[2] = alpha * alpha * grid['gcon_ks'][3,0]
    gamma = ucon_ks[0] * alpha

    velocity[0] = ucon_ks[1]/gamma + beta[0]/alpha
    velocity[1] = ucon_ks[2]/gamma + beta[1]/alpha
    velocity[2] = ucon_ks[3]/gamma + beta[2]/alpha
    soln['velocity'] = velocity

    """# compute magnetic 4-vector
    B = np.zeros((3, grid['n2'], grid['n1']), dtype=float)
    # radial magnetic field (B1 = 1/r^3)
    B[0,Ellipsis] = 1. / grid['r']**3

    gti    = grid['gcon'][1:4,0,Ellipsis]
    gij    = grid['gcov'][1:4,1:4,Ellipsis]
    beta_i = np.einsum('sji,ji->sji', gti, grid['lapse']**2)
    qsq    = np.einsum('yji,yji->ji', np.einsum('yxji,xji->yji', gij, utilde), utilde)
    gamma  = np.sqrt(1 + qsq)
    ui     = utilde - np.einsum('sji,ji->sji', beta_i, gamma/grid['lapse'])
    ut     = gamma/grid['lapse']

    bt = np.einsum('mji,mji->ji', np.einsum('msji,sji->mji', grid['gcov'][:,1:4,Ellipsis], B), ucon_mks)
    bi = (B + np.einsum('sji,ji->sji', ucon_mks[1:4,Ellipsis], bt)) / ucon_mks[None,0,Ellipsis]
    bcon_mks = np.append(bt[None,Ellipsis], bi, axis=0)
    bcov_mks = np.einsum('nmji,nji->mji', grid['gcov'], bcon_mks)

    soln['ucon'] = ucon_mks[:,0,:]
    soln['ucov'] = ucov_mks[:,0,:]
    soln['bcon'] = bcon_mks[:,0,:]
    soln['bcov'] = bcov_mks[:,0,:]
    soln['bsq']  = np.einsum('mi,mi->i', soln['bcon'], soln['bcov'])"""

def compute_vr_and_ut(r):
    gcov_bl(r)
    gcov_ks(r)
    gcon_ks()
    get_prim(r)
    compute_ub(r)
    return [soln['velocity'][0], soln['ucon_ks'][0]]



def model(uel, r):
    #idx = find_ind(np.exp(r), np.pi/2, 128, 128)[0]#this just finds the i index where the r at that
    #index is closests to the parameter r passed to model (so this is how I determine which
    # what the velocity and ut is that I should use)
    #There will be some error from the v_val and ut_val not being continuous, but it's blowing up really fast
    #print(idx/128 * 100, "percent done")
    v_ut = compute_vr_and_ut(r)
    v_val = v_ut[0]
    ut_val = v_ut[0]
    if (r < 0):
        print("uel: ", uel)
        print("r: ", r)
        print("v_val: ", v_val)
        print("ut_val: ", ut_val)
    m = 3.0
    dudr = -uel[0]*m/((r**(1.5))*ut_val*v_val)
    return dudr

def find_error():
    dump_kharma = fluid_dump.load_dump("./bondi.out0.{0:05}.phdf".format(0))
    r = dump_kharma['r'][:, 64, 0]
    #print("dump_kharma['r'].shape: ", dump_kharma['r'].shape)
    x1 = dump_kharma['X1'][:, 64, 0]
    """U1 = dump_kharma['u1'][:, 64, 0]
    U2 = dump_kharma['u2'][:, 64, 0]
    U3 = dump_kharma['u3'][:, 64, 0]
    gcov = dump_kharma['gcov']
    gcon = dump_kharma['gcon']
    lapse = dump_kharma['lapse'][:, 64, 0]
    qsq1 = gcov[1,1,:, 64, 0]*U1*U1+gcov[2,2,:, 64, 0]*U2*U2+gcov[3,3,:, 64, 0]*U3*U3
    qsq2 = gcov[1,2,:, 64, 0]*U1*U2+gcov[1,3,:, 64, 0]*U1*U3+gcov[2,3,:, 64, 0]*U2*U3
    qsq = qsq1+2*qsq2
    gamma = (1+qsq)**0.5
    ut = gamma/lapse
    ut_splrep = splrep(x1, ut)
    beta = lapse * lapse * gcon[1,0,Ellipsis][:,64,0]
    vr_splrep = splrep(x1, U1-beta/lapse*gamma)
    print("gamma: ", gamma)
    print("\n\n\n", "ut: ", ut)
"""

    load_data(True)

    dump_kharma2 = fluid_dump.load_dump("./bondi.out0.final.phdf")
    rho2 = dump_kharma2['rho'][:, 64, 0]
    kel2 = dump_kharma2['Kel_Howes'][:, 64, 0]
    game2 = 1.333333
    u_num = rho2**game2*kel2/(game2-1)
    u_ana = odeint(model, u_num[0], x1)
    print("u_num[127]: ", u_num[127])
    print("u_num[0]: ", u_num[0])
    print("u_ana[0]: ", u_ana[0])
    error = abs(u_num - u_ana)
    return [u_num, u_ana, error, r]

def plot():
    array = find_error()
    u_num = array[0]
    u_ana = array[1]
    errors = array[2]
    rs = array[3]
    """for i in range(128):
        u_num.append(array[0])
        u_ana.append(array[1])
    errors.append(array[2])
    rs.append(array[3])"""
    fig1 = plt.figure()
    plt.semilogy(rs, u_num, 'go', label = 'u_num')
    plt.semilogy(rs, u_ana, 'r.', label = 'u_ana')
    #plt.plot(rs, errors, 'b', label = 'difference between the two')
    plt.xlabel("radius")
    plt.title("potential energies")
    plt.ylabel("u")
    #plt.ylim(0, 5e-7)
    plt.legend()
    plt.savefig('uel_vs_r_128.png')
    plt.close()

def find_ind(r_want, th_want, itot, jtot):
    dump_kharma = fluid_dump.load_dump("./bondi.out0.00000.phdf")
    r_output = 0
    th_output = 64
    """th_output = 0
    thtemp = dump_kharma['th'][int(itot/2),0,0]
    for j in range (jtot):
        th = dump_kharma['th'][int(itot/2),j,0]
        if(abs(th-th_want)<abs(thtemp-th_want)):
            th_output = j
            thtemp = th"""
    rtemp = dump_kharma['r'][0,th_output,0]
    for i in range (itot):
        r = dump_kharma['r'][i,th_output,0]
        if(abs(r-r_want)<abs(rtemp-r_want)):
            r_output = i
            rtemp = r
    return [r_output, th_output]

if __name__=="__main__":
    r = 15
    th = np.pi/2
    """load_data(True)
    get_prim(r)
    gcov_bl(r,th)
    gcov_ks()
    gcon_ks()
    compute_ub()"""

    plot()
