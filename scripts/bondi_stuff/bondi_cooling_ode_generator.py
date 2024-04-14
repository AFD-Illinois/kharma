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

# python3 ./scripts/bondi_stuff/bondi_cooling_ode_generator.py
# this does assume we have a specific set of dump files and such

#analytic solution for part (next 8 functions):
dump = {}
grid = {}
soln = {}

def gcov_bl(r, th):
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
    grid['gcon_ks'] = np.linalg.inv(np.transpose(grid['gcov_ks'])) #fine because gcov_ks is symmetric

def load_data(res, read_grid=False):
    dfile = fluid_dump.load_dump("./dumps_{0:04}_fel5/bondi.out0.00000.phdf".format(res))
    dump['rc']    = dfile['rs']
    dump['mdot']  = dfile['mdot']
    dump['gam']   = dfile['gam']
    dump['rEH']   = dfile['r_eh'][()]
    grid['a'] = dfile['a']#I (Sam) added this here

    if read_grid:
        print("reading grid stuff")
        gfile  = fluid_dump.load_dump("./dumps_{0:04}_fel5/bondi.out0.00000.phdf".format(res))
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
            grid['a'] = dfile['a']#I'm using ks coords so IDK why gcov_bl and gcov_ks needs the "a" parameter, but I'll add it earlier too for now
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

    # We have u^r in BL. We need to convert this to ucon in MKS******but not for the electron cooling test
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
    dxdX[0,1] = 2*r / (r**2 - 2.*r + grid['a']**2)
    dxdX[3,1] = grid['a']/(r**2 - 2.*r + grid['a']**2)

    ucon_ks = np.zeros((4), dtype=float)
    for mu in range(4):
        for nu in range(4):
            ucon_ks[mu] += dxdX[mu,nu] * ucon_bl[nu]

    soln['ucon_ks'] = ucon_ks

    # Compute velocity primitives
    velocity = np.zeros((3), dtype=float)

    alpha = 1./np.sqrt(-grid['gcon_ks'][0,0])# gcon_ks is equivalent to gcon because the simulation was done in ks
    beta  = np.zeros((3), dtype=float)
    beta[0] = alpha * alpha * grid['gcon_ks'][0,1]
    beta[1] = alpha * alpha * grid['gcon_ks'][0,2]
    beta[2] = alpha * alpha * grid['gcon_ks'][0,3]
    gamma = ucon_ks[0] * alpha

    velocity[0] = ucon_ks[1]/gamma + beta[0]/alpha
    velocity[1] = ucon_ks[2]/gamma + beta[1]/alpha
    velocity[2] = ucon_ks[3]/gamma + beta[2]/alpha
    soln['velocity'] = ucon_ks[1]#velocity
    #I don't know why this^ works, but it does

def compute_vr_and_rho(r, th): #Sam's code
    gcov_bl(r, th)
    gcov_ks(r, th)
    gcon_ks()
    get_prim(r, th)
    compute_ub(r, th)

    return [soln['velocity'], soln['rho']]
#end of analytic solution stuff


#spline interpolation:
def find_splev(rs, ur, r):
    ur_splrep = splrep(rs, ur, k=3)
    return [splev(r, ur_splrep), splev(r, ur_splrep, der = 1)]

def dudr(uel, r, rs, ur):
    array = find_splev(rs, ur, r)
    ur_val = array[0]
    dvdr = array[1]
    x12_ur_derivitive = 2*r*ur_val + r*r*dvdr
    ur = array[0]
    game = 4.0/3.0

    compress = -game*uel*x12_ur_derivitive/(r*r*ur)
    cooling = -uel*3.0/((r**(1.5))*ur)
    return compress + cooling

def next_values(uel, r, rs, ur, h):
    check = False#this part checks if the spacing h is correct, because later in next_values I assume h is the spacing between point on the grid
    for i in rs:
        if(i == r):
            check = True
    if(check == False):
        print("your rk2 method is wrong!!!")

    k1 = dudr(uel, r, rs, ur)
    k2 = dudr(uel+h*k1/2, r+h/2, rs, ur)
    return uel + h*k2

def rk2_int(ulast, res, rs, ur, h):
    uels = [ulast]
    u_temp = ulast
    r_temp = rs[res-1]
    for i in range(res-1):
        u_temp = next_values(u_temp, r_temp, rs, ur, h)
        r_temp = r_temp + h
        uels.append(u_temp)
    return uels

def find_error(res):
    #this will find the r array with all r values:
    dump_kharma = fluid_dump.load_dump("./dumps_{0:04}/bondi.out0.00000.phdf".format(res))
    rs = dump_kharma['r'][:, 0, 0]
    th = dump_kharma['th'][:, 0, 0][0]
    h = rs[0]-rs[1]
    for i in range(4):
        rs = np.append(rs, rs[-1]-h)
        rs = np.insert(rs, 0, rs[0]+h)

    #to find the rho and ur arrays:
    load_data(res, True)
    ur = []
    rhos = []
    for r in rs:
        ur_rho_array = compute_vr_and_rho(r, th)#this returns an array of form [*, *]
        ur.append(ur_rho_array[0])
        rhos.append(ur_rho_array[1])
    print("rhos:", rhos)

    #for ulast:
    if (res==128):
        kel_last = 0.0478518540630092
    elif(res == 256):
        kel_last = 0.0480249227946548
    elif(res == 512):
        kel_last = 0.0481127265117538
    elif(res == 1024):
        kel_last = 0.0481569514223197
    ulast = rhos[-1]**game*kel_last/(game-1)

    #to find the analytical solution I will use rk4 with h = 1/2 the spacing of the gridzones in the simulation
    #I'm integrating in from the outer boundary so h will be negative and the inital value will be ulast
    u_ana = np.flip(rk2_int(ulast, res+8, rs, ur, h))
    kel_ana = []
    for i in range(res+8):
        kel_ana.append(u_ana[i]/rhos[i]**game*(game-1))

    return [kel_ana, rs]

def print_to_txt(res):
    array = find_error(res)[0]
    with open('soln_kel_'+ str(res) +'.txt', 'w') as file:
        for i in range(res + 8):
            file.write(str(array[i]))
            file.write("\n")

def plot_profile():
    #this gets the data:
    dump_kharma2 = fluid_dump.load_dump("./dumps_{0:04}/bondi.out0.final.phdf".format(1024))
    rho2 = dump_kharma2['rho'][:, 0, 0]
    kel2 = dump_kharma2['Kel_Howes'][:, 0, 0]
    u_num = rho2**game*kel2/(game-1) 
    rs2 = dump_kharma2['r'][:, 0, 0]
    rho = dump_kharma2['rho'][:, 0, 0]

    [u_ana128, rs128] = find_error(128)
    [u_ana256, rs256] = find_error(256)
    [u_ana512, rs512] = find_error(512)
    [u_ana1024, rs1024] = find_error(1024)

    #this plots the data for with cooling
    plt.semilogy(rs128, u_ana128, 'r,', label='128')
    plt.semilogy(rs256, u_ana256, 'g,', label='256')
    plt.semilogy(rs512, u_ana512, 'y,', label='512')
    plt.semilogy(rs1024, u_ana1024, 'k,', label='1024')
    plt.semilogy(rs2, kel2, 'b,', label='u_num')
    #plt.semilogy(rs, u_ana, 'r', label = 'rk2')

    #this makes the plot pretty
    plt.xlabel("radius")
    plt.title("uel profile at t=1000")
    plt.ylabel("uel")
    plt.legend()
    plt.savefig('quick_profile.png')
    plt.close()

if __name__=="__main__":
    game = 4/3

    #plot_profile()
    """print_to_txt(128)
    print_to_txt(256)
    print_to_txt(512)
    print_to_txt(1024)"""
    find_error(128)
    dump_kharma = fluid_dump.load_dump("./bondi.out0.00000.phdf".format(128))
    rhos_original = dump_kharma['rho'][:, 0, 0]
    print('rhos actual:', rhos_original)