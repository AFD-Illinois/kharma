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

# python3 ./scripts/bondi_stuff/less_weird_rk2_for_cooling_conv.py
# right now, it just creates a graph that shows the numerical and anylitical solution rather than a convergence plot

def r_to_index(res, r_val):
    dump_kharma = fluid_dump.load_dump("./dumps_{0:05}_without_cooling/bondi.out0.00000.phdf".format(res))
    rs = dump_kharma['r'][:,0,0]
    index = 0
    temp = 0
    for i in range(res):
        if(abs(rs[i]-r_val) < abs(temp-r_val)):
            temp = rs[i]
            index = i
    return index

def index_to_r(res, ind):
    dump_kharma = fluid_dump.load_dump("./dumps_{0:05}_without_cooling/bondi.out0.00000.phdf".format(res))
    return dump_kharma['r'][ind,0,0]

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
    dump_kharma = fluid_dump.load_dump("./dumps_{0:04}_fel5/bondi.out0.00000.phdf".format(res))
    rs = dump_kharma['r'][:, 0, 0]

    #to find the value of uel and r at the center of the first outter ghost zone:
    if(res == 128):
        ulast = u_last_128
        rlast = r_last_128
        urlast = ur_last_128
    elif(res == 256):
        ulast = u_last_256
        rlast = r_last_256
        urlast = ur_last_256
    elif(res == 512):
        ulast = u_last_512
        rlast = r_last_512
        urlast = ur_last_512
    elif(res == 1024):
        ulast = u_last_1024
        rlast = r_last_1024
        urlast = ur_last_1024

    #to find ur
    U1 = dump_kharma['u1'][:,0,0]
    U2 = dump_kharma['u2'][:,0,0]
    U3 = dump_kharma['u3'][:,0,0]
    gcov = dump_kharma['gcov']
    lapse = dump_kharma['lapse'][:,0,0]
    gti = dump_kharma['gcon']
    beta = gti[1,0,:,0,0]*lapse*lapse
    qsq1 = gcov[1,1,:,0,0]*U1*U1+gcov[2,2,:,0,0]*U2*U2+gcov[3,3,:,0,0]*U3*U3
    qsq2 = gcov[1,2,:,0,0]*U1*U2+gcov[1,3,:,0,0]*U1*U3+gcov[2,3,:,0,0]*U2*U3
    qsq = qsq1+2*qsq2
    gamma = (1+qsq)**0.5
    ur = U1 - gamma*beta/lapse
    ur = np.append(ur, urlast)#from the ghost zone
    rs = np.append(rs, rlast)

    #to find the numerical solution
    dump_kharma2 = fluid_dump.load_dump("./dumps_{0:04}_fel5/bondi.out0.final.phdf".format(res))
    rho2 = dump_kharma2['rho'][:, 0, 0]
    kel2 = dump_kharma2['Kel_Howes'][:, 0, 0]
    u_num = rho2**game*kel2/(game-1) 

    #to find the analytical solution I will use rk4 with h = 1/2 the spacing of the gridzones in the simulation
    #I'm integrating in from the outer boundary so h will be negative and the inital value will be ulast
    h = rs[0]-rs[1]
    u_ana = np.flip(rk2_int(ulast, res+1, rs, ur, h))
    if(res==128):
        print("ulast:", ulast)
        print("u_ana length:", len(u_ana))
        print(u_ana)
        print("rlast:", rs[-1])
        print("rs length:", len(rs))
        print(rs)

    rs = rs[:-1]#this gets rid of the ghost zone data
    u_ana = u_ana[:-1]#this gets rid of the ghost zone data

    if (res==128):
        print("len(ur):", len(ur))

    #to find the error between the two solutions
    error = abs(u_num - u_ana)

    return [u_num, u_ana, error, rs]


def find_error_no_ghost(res):
    dump_kharma = fluid_dump.load_dump("./dumps_{0:04}_double/bondi.out0.00000.phdf".format(res))
    rs = dump_kharma['r'][:, 0, 0]

    #to find ur
    U1 = dump_kharma['u1'][:,0,0]
    U2 = dump_kharma['u2'][:,0,0]
    U3 = dump_kharma['u3'][:,0,0]
    gcov = dump_kharma['gcov']
    lapse = dump_kharma['lapse'][:,0,0]
    gti = dump_kharma['gcon']
    beta = gti[1,0,:,0,0]*lapse*lapse
    qsq1 = gcov[1,1,:,0,0]*U1*U1+gcov[2,2,:,0,0]*U2*U2+gcov[3,3,:,0,0]*U3*U3
    qsq2 = gcov[1,2,:,0,0]*U1*U2+gcov[1,3,:,0,0]*U1*U3+gcov[2,3,:,0,0]*U2*U3
    qsq = qsq1+2*qsq2
    gamma = (1+qsq)**0.5
    ur = U1 - gamma*beta/lapse

    #to find the numerical solution
    dump_kharma2 = fluid_dump.load_dump("./dumps_{0:04}_double/bondi.out0.final.phdf".format(res))
    rho2 = dump_kharma2['rho'][:, 0, 0]
    kel2 = dump_kharma2['Kel_Howes'][:, 0, 0]
    u_num = rho2**game*kel2/(game-1) 

    #to find the analytical solution I will use rk4 with h = 1/2 the spacing of the gridzones in the simulation
    #I'm integrating in from the outer boundary so h will be negative and the inital value will be ulast
    h = rs[0]-rs[1]
    u_ana = np.flip(rk2_int(u_num[-1], res, rs, ur, h))

    #to find the error between the two solutions
    error = abs(u_num - u_ana)

    return [u_num, u_ana, error, rs]

def find_error_first_dump(res):
    dump_kharma = fluid_dump.load_dump("./dumps_{0:04}_double/bondi.out0.00000.phdf".format(res))
    rs = dump_kharma['r'][:, 0, 0]
    initial_kel = dump_kharma['Kel_Howes'][res - 1, 0, 0]
    initial_rho = dump_kharma['rho'][res - 1, 0, 0]
    ulast_init = initial_rho**game*initial_kel/(game-1)

    #to find ur
    U1 = dump_kharma['u1'][:,0,0]
    U2 = dump_kharma['u2'][:,0,0]
    U3 = dump_kharma['u3'][:,0,0]
    gcov = dump_kharma['gcov']
    lapse = dump_kharma['lapse'][:,0,0]
    gti = dump_kharma['gcon']
    beta = gti[1,0,:,0,0]*lapse*lapse
    qsq1 = gcov[1,1,:,0,0]*U1*U1+gcov[2,2,:,0,0]*U2*U2+gcov[3,3,:,0,0]*U3*U3
    qsq2 = gcov[1,2,:,0,0]*U1*U2+gcov[1,3,:,0,0]*U1*U3+gcov[2,3,:,0,0]*U2*U3
    qsq = qsq1+2*qsq2
    gamma = (1+qsq)**0.5
    ur = U1 - gamma*beta/lapse

    #to find the numerical solution
    dump_kharma2 = fluid_dump.load_dump("./dumps_{0:04}_double/bondi.out0.final.phdf".format(res))
    rho2 = dump_kharma2['rho'][:, 0, 0]
    kel2 = dump_kharma2['Kel_Howes'][:, 0, 0]
    u_num = rho2**game*kel2/(game-1) 



    #to find the analytical solution I will use rk4 with h = 1/2 the spacing of the gridzones in the simulation
    #I'm integrating in from the outer boundary so h will be negative and the inital value will be ulast
    h = rs[0]-rs[1]
    u_ana = np.flip(rk2_int(ulast_init, res, rs, ur, h))

    #to find the error between the two solutions
    error = abs(u_num - u_ana)

    return [u_num, u_ana, error, rs]


def plot_profile():
    #this gets the data:
    dump_kharma = fluid_dump.load_dump("./bondi.out0.00000.phdf")
    rho = dump_kharma['rho'][:, 0, 0]
    kel = dump_kharma['Kel_Howes'][:, 0, 0]
    u_num = rho**game*kel/(game-1) 
    rs = dump_kharma['r'][:, 0, 0]

    dump_kharma2 = fluid_dump.load_dump("./bondi.out0.00024.phdf")
    rho2 = dump_kharma2['rho'][:, 0, 0]
    kel2 = dump_kharma2['Kel_Howes'][:, 0, 0]
    u_num2 = rho2**game*kel2/(game-1)

    #this plots the data for with cooling
    plt.semilogy(rs, kel, 'r', label='initial')
    plt.semilogy(rs, kel2, 'b', label = 'final')
    #plt.semilogy(rs, rho, 'r')
    #plt.semilogy(rs, u_ana, 'r', label = 'rk2')

    #this makes the plot pretty
    plt.xlabel("radius")
    plt.title("uel profile at t=0 and t=1000")
    plt.ylabel("uel")
    plt.legend()
    plt.savefig('quick_profile.png')
    plt.close()

def plot_error_profile():
    #this gets the data:
    arraywith = find_error(128)
    errorswith = arraywith[2]
    rs = arraywith[3]

    arraywithout = find_error_no_ghost(128)
    errorswithout = arraywithout[2]

    arrayinit = find_error_first_dump(128)
    errorsinit = arrayinit[2]

    #this plots the data for with cooling
    plt.semilogy(rs, errorswith, 'b', label = 'ghost zone data')
    plt.semilogy(rs, errorswithout, 'r', label = 'last gridzone of last dump')
    plt.semilogy(rs, errorsinit, 'g', label = 'last gridzone of 0th dump')

    #this makes the plot pretty
    plt.xlabel("radius")
    plt.title("error in uel using different initial data")
    plt.ylabel("fractional difference between kharma truncation and rk2")
    plt.legend()
    plt.savefig('less_weird_error_profiles.png')
    plt.close()


def plot_errors():
    #this gets the data:
    array128 = find_error(128)
    errors128 = array128[2]
    array256 = find_error(256)
    errors256 = array256[2]
    array512 = find_error(512)
    errors512 = array512[2]
    array1024 = find_error(1024)
    errors1024 = array1024[2]

    r_for_ind = 15
    ind128 = r_to_index(128, r_for_ind)
    ind256 = r_to_index(256, r_for_ind)
    ind512 = r_to_index(512, r_for_ind)
    ind1024 = r_to_index(1024, r_for_ind)

    print(ind128)
    print(ind256)
    print(ind512)
    print(ind1024)

    #this plots the data for with cooling
    plt.loglog([128.0, 256.0, 512.0, 1024.0], [errors128[ind128], errors256[ind256], errors512[ind512], errors1024[ind1024]], 'b', label = 'error')
    plt.loglog([128.0, 256.0, 512.0, 1024.0], [errors128[ind128], errors256[ind256], errors512[ind512], errors1024[ind1024]], 'bo')
    plt.loglog([128.0, 256.0, 512.0, 1024.0], [errors128[ind128], errors128[ind128]*2**(-2), errors128[ind128]*4**(-2), errors128[ind128]*8**(-2)] , 'r', label = 'N^-2')
    plt.loglog([128.0, 256.0, 512.0, 1024.0], [errors128[ind128], errors128[ind128]*2**(-1), errors128[ind128]*4**(-1), errors128[ind128]*8**(-1)] , 'g', label = 'N^-1')

    #this makes the plot pretty
    plt.xlabel("resolution")
    plt.title("error in uel vs resolution with cooling")
    plt.ylabel("error in internal energy of the electrons")
    plt.legend()
    plt.savefig('less_weird_electrons_conv_rk2.png')
    plt.close()

if __name__=="__main__":
    #data for different things at the first outter ghost zone (I found these just by printing them out durring a kharma run):
    game = 4/3

    r_last_128 = 30.1054687500000000
    u_last_128_old = 0.0000003867018181
    kel_last_128=0.0481506202956166#0.0009630124059123
    rho_last_128=0.0012444209330380
    ur_last_128 =-0.0705556895512476
    u_last_128 = rho_last_128**game*kel_last_128/(game-1)

    r_last_256 = 30.0527343750000000
    u_last_256_old = 0.0000003877208771
    kel_last_256=0.0481759714664538#0.0009635194293291
    rho_last_256=0.0012463875188361
    ur_last_256 = -0.0706918030748616
    u_last_256 = rho_last_256**game*kel_last_256/(game-1)

    r_last_512 = 30.0263671875000000
    u_last_512_old = 0.0000003882322876
    kel_last_512=0.0481886737649923#0.0009637734752998
    rho_last_512=0.0012473736626378
    ur_last_512 = -0.0707600259563606
    u_last_512 = rho_last_512**game*kel_last_512/(game-1)

    r_last_1024 = 30.0131835937500000
    u_last_1024_old = 0.0000003884884650
    kel_last_1024=0.0481950316091551#0.0009639006321831
    rho_last_1024=0.0012478674495709
    ur_last_1024 = -0.0707941790460781
    u_last_1024 = rho_last_1024**game*kel_last_1024/(game-1)


    plot_profile()