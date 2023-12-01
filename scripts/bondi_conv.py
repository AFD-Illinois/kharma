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
# right now, it just creates a graph that shows the numerical and anylitical solution rather than a convergence plot
dump = {}
grid = {}
soln = {}


def gcov_bl(r, th):
    print(grid['gcov'].shape)
    grid['gcov_bl'] = np.zeros_like(grid['gcov'][:,:,0])

    DD = 1 - 2./r + grid['a']**2/r**2
    mu = 1 + grid['a']**2 * np.cos(th)**2 / r**2

    grid['gcov_bl'][0,0] = -(1 - 2./(r * mu))
    grid['gcov_bl'][0,3] = -2 * grid['a'] * np.sin(th)**2 / (r * mu)
    grid['gcov_bl'][3,0] = grid['gcov_bl'][0,3]
    grid['gcov_bl'][1,1] = mu / DD
    grid['gcov_bl'][2,2] = r**2 * mu
    grid['gcov_bl'][3,3] = r**2 * np.sin(th)**2 * (1 + grid['a']**2/r**2 \
                                    + 2 * grid['a']**2 * np.sin(th)**2 / (r**3 * mu))

def gcov_ks(r, th):
    grid['gcov_ks'] = np.zeros_like(grid['gcov'][:,:,0])
    sigma = r**2 + (grid['a']**2 * np.cos(th)**2)
    
    grid['gcov_ks'][0,0] = -1 + 2*r/sigma
    grid['gcov_ks'][0,1] = 2*r/sigma
    grid['gcov_ks'][0,3] = -(2*grid['a']*r*np.sin(th)**2)/sigma
    grid['gcov_ks'][1,0] = 2*r/sigma
    grid['gcov_ks'][1,1] = 1 + 2*r/sigma
    grid['gcov_ks'][1,3] = -grid['a']*np.sin(th)**2 * (1 + 2*r/sigma)
    grid['gcov_ks'][2,2] = sigma
    grid['gcov_ks'][3,0] = -(2*grid['a']*r*np.sin(th)**2)/sigma
    grid['gcov_ks'][3,1] = -grid['a']*np.sin(th)**2 * (1 + 2*r/sigma)
    grid['gcov_ks'][3,3] = np.sin(th)**2 * (sigma + grid['a']**2*np.sin(th)**2 * (1 + 2*r/sigma))

def gcon_ks():
    grid['gcon_ks'] = np.linalg.inv(np.transpose(grid['gcov_ks']))

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

        grid['rEH_ind'] = np.argmin(np.fabs(grid['r']-dump['rEH']) > 0.)
        grid['n1']  = 128
        grid['n2']  = 1
        grid['n3']  = 1
        grid['dx1'] = dfile['dx1']
        grid['dx2'] = dfile['dx2']

        grid['x1'] = np.squeeze(gfile['X1'][()])
        grid['x2'] = np.squeeze(gfile['X2'][()])
        grid['x3'] = np.squeeze(gfile['X3'][()])

        #I'm pretty sure this is what I'm supposed to do for grid['metric']
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

def get_prim(x1, th):
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
        print("Not converged at r = {:.2f}", x1)

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
    dxdX[1,3] = grid['a']/(r**2 - 2.*r + grid['a']**2)#fliped 1 and 3 (I don't think it actually matters as long as I'm consistant)

    ucon_ks = np.zeros((4), dtype=float)
    for mu in range(4):
        for nu in range(4):
            ucon_ks[mu] += dxdX[nu,mu] * ucon_bl[nu]

    soln['ucon_ks'] = ucon_ks

    # Compute velocity primitives
    velocity = np.zeros((3), dtype=float)

    alpha = 1./np.sqrt(-grid['gcon_ks'][0,0])# should these be gcon instead of gcon_ks? I don't think so because gcon is 4x4x128
    beta  = np.zeros((3), dtype=float)
    beta[0] = alpha * alpha * grid['gcon_ks'][1,0]
    beta[1] = alpha * alpha * grid['gcon_ks'][2,0]
    beta[2] = alpha * alpha * grid['gcon_ks'][3,0]
    gamma = ucon_ks[0] * alpha

    velocity[0] = ucon_ks[1]/gamma + beta[0]/alpha
    velocity[1] = ucon_ks[2]/gamma + beta[1]/alpha
    velocity[2] = ucon_ks[3]/gamma + beta[2]/alpha
    soln['velocity'] = velocity

def compute_vr_and_ut(r, th): #Sam's code
    gcov_bl(r, th)
    gcov_ks(r, th)
    gcon_ks()
    get_prim(r, th)
    compute_ub(r, th)
    return [soln['velocity'][0], soln['ucon_ks'][0]]

def model(uel, r): # Sam's code
    #idx = find_ind(np.exp(r), np.pi/2, 128, 128)[0]#this just finds the i index where the r at that
    #index is closests to the parameter r passed to model (so this is how I determine which
    # what the velocity and ut is that I should use)
    #There will be some error from the v_val and ut_val not being continuous, but it's blowing up really fast
    #print(idx/128 * 100, "percent done")
    v_ut = compute_vr_and_ut(r, np.pi/2)
    v_val = v_ut[0]
    ut_val = v_ut[1]
    """if (r < 0):
        print("uel: ", uel)
        print("r: ", r)
        print("v_val: ", v_val)
        print("ut_val: ", ut_val)"""
    m = 3.0
    dudr = -uel[0]*m/((r**(1.5))*ut_val*v_val)
    return dudr

def find_error(): #Sam's code
    dump_kharma = fluid_dump.load_dump("./bondi.out0.{0:05}.phdf".format(0))
    lastkel = dump_kharma['Kel_Howes'][127, 0, 0]
    lastrho = dump_kharma['rho'][127, 0, 0]
    lastgame = 1.333333
    ulast = lastrho**lastgame*lastkel/(lastgame-1)
    r = dump_kharma['r'][:, 0, 0]
    #print("dump_kharma['r'].shape: ", dump_kharma['r'].shape)
    x1 = dump_kharma['X1'][:, 0, 0]

    load_data(True)

    dump_kharma2 = fluid_dump.load_dump("./bondi.out0.final.phdf")
    rho2 = dump_kharma2['rho'][:, 0, 0]
    kel2 = dump_kharma2['Kel_Howes'][:, 0, 0]
    game2 = 1.333333
    u_num = rho2**game2*kel2/(game2-1)
    u_ana = np.flip(odeint(model, ulast, np.flip(x1)))
    print("u_num[127]: ", u_num[127])
    print("u_num[0]: ", u_num[0])
    print("u_ana[127]: ", u_ana[127])
    print("u_ana[0]: ", u_ana[0])
    error = abs(u_num - u_ana)
    return [u_num, u_ana, error, x1]

def plot(): #Sam's code
    array = find_error()
    u_num = array[0]
    u_ana = array[1]
    errors = array[2]
    rs = array[3]
    fig1 = plt.figure()
    print("plotting...")
    plt.plot(rs, u_num, 'go', label = 'u_num')
    plt.plot(rs, u_ana, 'r.', label = 'u_ana')
    #plt.plot(rs, errors, 'b', label = 'difference between the two')
    plt.xlabel("radius")
    plt.title("potential energies")
    plt.ylabel("u")
    #plt.ylim(0, 5e-7)
    plt.legend()
    plt.savefig('uel_vs_r.png')
    plt.close()
    print("finished plotting")

if __name__=="__main__":
    plot()
