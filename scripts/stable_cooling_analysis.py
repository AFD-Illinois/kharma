import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprocessing as mp
import pyharm.coordinates as coords
import pyharm.fluid_dump as fluid_dump
import sys
sys.path.append('/home/samason4/kharma_cooling/pyharm')

# python3 ./scripts/stable_cooling_analysis.py

def uel_vs_t(i, j, dumpno):
    #numerical:
    dump_kharma = fluid_dump.load_dump("./torus.out0.{0:05}.phdf".format(0))
    U1 = dump_kharma['u1'][i,j,0]
    U2 = dump_kharma['u2'][i,j,0]
    U3 = dump_kharma['u3'][i,j,0]
    rho = dump_kharma['rho'][i,j,0]
    gcov = dump_kharma['gcov']
    lapse = dump_kharma['lapse'][i,j,0]
    kel = dump_kharma['Kel_Howes'][i,j,0]
    game = 1.333333
    qsq1 = gcov[1,1,i,j,0]*U1*U1+gcov[2,2,i,j,0]*U2*U2+gcov[3,3,i,j,0]*U3*U3
    qsq2 = gcov[1,2,i,j,0]*U1*U2+gcov[1,3,i,j,0]*U1*U3+gcov[2,3,i,j,0]*U2*U3
    qsq = qsq1+2*qsq2
    gamma = (1+qsq)**0.5
    ut = gamma/lapse
    r = dump_kharma['r'][i,j,0]
    m = 3.0
    alpha = -1*r**(-3/2)*m/ut
    u_init = rho**game*kel/(game-1)

    dump_kharma2 = fluid_dump.load_dump("./torus.out0.{0:05}.phdf".format(dumpno))
    rho2 = dump_kharma2['rho'][i,j,0]
    kel2 = dump_kharma2['Kel_Howes'][i,j,0]
    game2 = 1.333333
    t = dump_kharma2['t']
    u_num = rho2**game2*kel2/(game2-1)
    u_ana = u_init*np.exp(alpha*float(t))
    return [u_num, u_ana, t]

def plot_uel(i, j):
    u_ana_arr = []
    u_num_arr = []
    t_arr = []
    for dumpno in range (19):
        temp = uel_vs_t(i, j, dumpno)
        u_num_arr.append(temp[0])
        u_ana_arr.append(temp[1])
        t_arr.append(temp[2])
        print(dumpno*5, "%")
    fig1 = plt.figure()
    plt.plot(t_arr, u_ana_arr, 'bo')
    plt.plot(t_arr, u_num_arr, 'r.')
    plt.savefig('u_vs_t.png')
    plt.close()

def plot_conv_res(r_want, th_want):
    indicies128 = find_ind(128, r_want, th_want, 128, 128)
    indicies256 = find_ind(256, r_want, th_want, 256, 256)
    u128 = find_error128(5, indicies128[0], indicies128[1])[0]
    u256 = find_error256(5, indicies256[0], indicies256[1])[0]
    fig1, sub1 = plt.subplots()
    plt.loglog([128, 256], [u128, u256], 'b', label = 'numerical error')
    plt.loglog([128, 256], [u128+1e-9, u128*(2)**(-2.0)+1e-9], 'r', label = 'line of slope N^-2')
    plt.xticks([], [])
    sub1.set_xticks([])
    sub1.set_xticks([], minor=True)
    sub1.set_xticks([128,256])
    sub1.set_xticklabels(['128','256'])
    plt.xlabel("resolution (the simulations were nxnx1)")
    plt.ylabel("total error in internal energy")
    plt.title("error vs resolution at r=10, theta=pi/2, and t=20")
    plt.legend()
    plt.savefig('error_conv_res.png')
    plt.close()

def plot_error_res(r_want, th_want):
    indicies256 = find_ind(256, r_want, th_want, 256, 256)
    print("got indicies")
    u256 = []
    tarr256 = []
    for i in range(19):
        temp1 = find_error256(i, indicies256[0], indicies256[1])
        u256.append(temp1[0])
        tarr256.append(temp1[1])
        print("working on errors 1")
    indicies128 = find_ind(128, r_want, th_want, 128, 128)
    print("got indicies")
    u128 = []
    tarr128 = []
    for i in range(19):
        temp2 = find_error128(i, indicies128[0], indicies128[1])
        u128.append(temp2[0])
        tarr128.append(temp2[1])
        print("working on errors 2")
    plt.plot(tarr256, u256, 'b')
    plt.plot(tarr256, u256, 'bo', label = '256x256x1 resolution')
    plt.plot(tarr128, u128, 'r')
    plt.plot(tarr128, u128, 'r.', label = '128x128x1 resolution')
    combined = []
    tcomb = []
    for i in range(19):
        combined.append(abs(u128[i]-u256[i]))
        tcomb.append((tarr128[i]+tarr256[i])/2)
    plt.plot(tcomb, combined, 'g')
    plt.plot(tcomb, combined, 'g.', label = 'difference between the two')
    plt.xlabel("time")
    plt.ylabel("total error in internal energy")
    plt.title("error vs resolution vs t")
    plt.legend()
    plt.savefig('error_vs_t_res.png')
    plt.close()

def plot_error_cour(r_want, th_want):
    indicies = find_ind(9, r_want, th_want, 128, 128)
    print("got indicies")
    u9 = []
    tarr9 = []
    for i in range(19):
        temp1 = find_error9(i, indicies[0], indicies[1])
        u9.append(temp1[0])
        tarr9.append(temp1[1])
        print("working on errors 1")
    u5 = []
    tarr5 = []
    for i in range(19):
        temp2 = find_error128(i, indicies[0], indicies[1])
        u5.append(temp2[0])
        tarr5.append(temp2[1])
        print("working on errors 2")
    plt.plot(tarr9, u9, 'b')
    plt.plot(tarr9, u9, 'bo', label = 'courant number = 0.9')
    plt.plot(tarr5, u5, 'r')
    plt.plot(tarr5, u5, 'r.', label = 'courant number = 0.5')
    combined = []
    tcomb = []
    for i in range(19):
        combined.append(abs(u9[i]-u5[i]))
        tcomb.append((tarr9[i]+tarr5[i])/2)
    plt.plot(tcomb, combined, 'g')
    plt.plot(tcomb, combined, 'g.', label = 'difference between the two')
    plt.xlabel("time")
    plt.ylabel("total error in internal energy")
    plt.title("error vs courant number vs t")
    plt.legend()
    plt.savefig('error_vs_t_cour.png')
    plt.close()

def find_error256(dumpno, i, j):
    dump_kharma = fluid_dump.load_dump("./dumps256/torus.out0.{0:05}.phdf".format(0))
    U1 = dump_kharma['u1'][i,j,0]
    U2 = dump_kharma['u2'][i,j,0]
    U3 = dump_kharma['u3'][i,j,0]
    rho = dump_kharma['rho'][i,j,0]
    gcov = dump_kharma['gcov']
    lapse = dump_kharma['lapse'][i,j,0]
    kel = dump_kharma['Kel_Howes'][i,j,0]
    game = 1.333333
    qsq1 = gcov[1,1,i,j,0]*U1*U1+gcov[2,2,i,j,0]*U2*U2+gcov[3,3,i,j,0]*U3*U3
    qsq2 = gcov[1,2,i,j,0]*U1*U2+gcov[1,3,i,j,0]*U1*U3+gcov[2,3,i,j,0]*U2*U3
    qsq = qsq1+2*qsq2
    gamma = (1+qsq)**0.5
    ut = gamma/lapse
    r = dump_kharma['r'][i,j,0]
    m = 3.0
    alpha = -1*r**(-3/2)*m/ut
    u_init = rho**game*kel/(game-1)

    dump_kharma2 = fluid_dump.load_dump("./dumps256/torus.out0.{0:05}.phdf".format(dumpno))
    rho2 = dump_kharma2['rho'][i,j,0]
    kel2 = dump_kharma2['Kel_Howes'][i,j,0]
    game2 = 1.333333
    t = dump_kharma2['t']
    u_num = rho2**game2*kel2/(game2-1)
    u_ana = u_init*np.exp(alpha*float(t))
    return [abs(u_num-u_ana), t]

def find_error128(dumpno, i, j):
    dump_kharma = fluid_dump.load_dump("./dumps128/torus.out0.{0:05}.phdf".format(0))
    U1 = dump_kharma['u1'][i,j,0]
    U2 = dump_kharma['u2'][i,j,0]
    U3 = dump_kharma['u3'][i,j,0]
    rho = dump_kharma['rho'][i,j,0]
    gcov = dump_kharma['gcov']
    lapse = dump_kharma['lapse'][i,j,0]
    kel = dump_kharma['Kel_Howes'][i,j,0]
    game = 1.333333
    qsq1 = gcov[1,1,i,j,0]*U1*U1+gcov[2,2,i,j,0]*U2*U2+gcov[3,3,i,j,0]*U3*U3
    qsq2 = gcov[1,2,i,j,0]*U1*U2+gcov[1,3,i,j,0]*U1*U3+gcov[2,3,i,j,0]*U2*U3
    qsq = qsq1+2*qsq2
    gamma = (1+qsq)**0.5
    ut = gamma/lapse
    r = dump_kharma['r'][i,j,0]
    m = 3.0
    alpha = -1*r**(-3/2)*m/ut
    u_init = rho**game*kel/(game-1)

    dump_kharma2 = fluid_dump.load_dump("./dumps128/torus.out0.{0:05}.phdf".format(dumpno))
    rho2 = dump_kharma2['rho'][i,j,0]
    kel2 = dump_kharma2['Kel_Howes'][i,j,0]
    game2 = 1.333333
    t = dump_kharma2['t']
    u_num = rho2**game2*kel2/(game2-1)
    u_ana = u_init*np.exp(alpha*float(t))
    return [abs(u_num-u_ana), t]

def find_error9(dumpno, i, j):
    dump_kharma = fluid_dump.load_dump("./dumps9/torus.out0.{0:05}.phdf".format(0))
    U1 = dump_kharma['u1'][i,j,0]
    U2 = dump_kharma['u2'][i,j,0]
    U3 = dump_kharma['u3'][i,j,0]
    rho = dump_kharma['rho'][i,j,0]
    gcov = dump_kharma['gcov']
    lapse = dump_kharma['lapse'][i,j,0]
    kel = dump_kharma['Kel_Howes'][i,j,0]
    game = 1.333333
    qsq1 = gcov[1,1,i,j,0]*U1*U1+gcov[2,2,i,j,0]*U2*U2+gcov[3,3,i,j,0]*U3*U3
    qsq2 = gcov[1,2,i,j,0]*U1*U2+gcov[1,3,i,j,0]*U1*U3+gcov[2,3,i,j,0]*U2*U3
    qsq = qsq1+2*qsq2
    gamma = (1+qsq)**0.5
    ut = gamma/lapse
    r = dump_kharma['r'][i,j,0]
    m = 3.0
    alpha = -1*r**(-3/2)*m/ut
    u_init = rho**game*kel/(game-1)

    dump_kharma2 = fluid_dump.load_dump("./dumps9/torus.out0.{0:05}.phdf".format(dumpno))
    rho2 = dump_kharma2['rho'][i,j,0]
    kel2 = dump_kharma2['Kel_Howes'][i,j,0]
    game2 = 1.333333
    t = dump_kharma2['t']
    u_num = rho2**game2*kel2/(game2-1)
    u_ana = u_init*np.exp(alpha*float(t))
    return [abs(u_num-u_ana), t]

def find_error5(dumpno, i, j):
    dump_kharma = fluid_dump.load_dump("./dumps5/torus.out0.{0:05}.phdf".format(0))
    U1 = dump_kharma['u1'][i,j,0]
    U2 = dump_kharma['u2'][i,j,0]
    U3 = dump_kharma['u3'][i,j,0]
    rho = dump_kharma['rho'][i,j,0]
    gcov = dump_kharma['gcov']
    lapse = dump_kharma['lapse'][i,j,0]
    kel = dump_kharma['Kel_Howes'][i,j,0]
    game = 1.333333
    qsq1 = gcov[1,1,i,j,0]*U1*U1+gcov[2,2,i,j,0]*U2*U2+gcov[3,3,i,j,0]*U3*U3
    qsq2 = gcov[1,2,i,j,0]*U1*U2+gcov[1,3,i,j,0]*U1*U3+gcov[2,3,i,j,0]*U2*U3
    qsq = qsq1+2*qsq2
    gamma = (1+qsq)**0.5
    ut = gamma/lapse
    r = dump_kharma['r'][i,j,0]
    m = 3.0
    alpha = -1*r**(-3/2)*m/ut
    u_init = rho**game*kel/(game-1)

    dump_kharma2 = fluid_dump.load_dump("./dumps5/torus.out0.{0:05}.phdf".format(dumpno))
    rho2 = dump_kharma2['rho'][i,j,0]
    kel2 = dump_kharma2['Kel_Howes'][i,j,0]
    game2 = 1.333333
    t = dump_kharma2['t']
    u_num = rho2**game2*kel2/(game2-1)
    u_ana = u_init*np.exp(alpha*float(t))
    return [abs(u_num-u_ana), t]

def plot_conv_cour(r_want, th_want):
    indicies = find_ind(9, r_want, th_want, 128, 128)
    
    u9 = find_error9(5, indicies[0], indicies[1])[0]
    u5 = find_error5(5, indicies[0], indicies[1])[0]
    fig1, sub1 = plt.subplots()
    plt.loglog([1/0.9, 1/0.5], [u9, u5], 'b', label = 'numerical error')
    plt.loglog([1/0.9, 1/0.5], [u9, u5], 'b.')
    plt.loglog([1/0.9, 1/0.5], [u9+1e-9, u9*(2)**(-2.0)+1e-9], 'r', label = 'line of slope N^-2')
    plt.xticks([], [])
    sub1.set_xticks([])
    sub1.set_xticks([], minor=True)
    sub1.set_xticks([1/0.9,1/0.5])
    sub1.set_xticklabels(['1/0.9','1/0.5'])
    plt.xlabel("1/(courant number)")
    plt.ylabel("total error in internal energy")
    plt.title("error vs courant at r=10, theta=pi/2, and t=20")
    plt.legend()
    plt.savefig('error_conv_cour.png')
    plt.close()

def find_ind(dumpsdir, r_want, th_want, itot, jtot):
    dump_kharma = fluid_dump.load_dump("./dumps{0:01}/torus.out0.00000.phdf".format(dumpsdir))
    r_output = 0
    th_output = 0
    thtemp = dump_kharma['th'][int(itot/2),0,0]
    for j in range (jtot):
        th = dump_kharma['th'][int(itot/2),j,0]
        if(abs(th-th_want)<abs(thtemp-th_want)):
            th_output = j
            thtemp = th
    rtemp = dump_kharma['r'][0,th_output,0]
    for i in range (itot):
        r = dump_kharma['r'][i,th_output,0]
        if(abs(r-r_want)<abs(rtemp-r_want)):
            r_output = i
            rtemp = r
    return [r_output, th_output]

if __name__=="__main__":
    """
    #getting errors:
    errors = []
    cour_inv = []
    errors9 = find_error(9, 50, 50)
    errors5 = find_error(5, 50, 50)
    #errors2 = find_error(2, 5, 5)
    errors.append(errors9)
    cour_inv.append(1/.9)
    errors.append(errors5)
    cour_inv.append(1/.5)
    #errors.append(errors2)
    #cour_inv.append(1/.2)

    #this is for the comparison line:
    x = []
    res = []
    x2 = []
    res2 = []
    temp_res = 1
    temp_x = .5e-18
    for i in range(2):
        x.append(temp_x*temp_res**(-2))
        res.append(temp_res)
        temp_res += 2
    temp_res2 = 1
    temp_x2 = .5e-18
    for i in range(2):
        x2.append(temp_x2*temp_res2**(-1))
        res2.append(temp_res2)
        temp_res2 += 2

    #plotting:
    fig1, sub1 = plt.subplots()
    sub1.loglog(cour_inv, errors, color = 'b', label = 'Error of numerical testcooling')
    sub1.loglog(res, x, color = 'r', label = 'Line of Slope N^-2 for Comparison')
    sub1.loglog(res2, x2, color = 'g', label = 'Line of Slope N^-1 for Comparison')
    sub1.loglog(cour_inv, errors, 'bo')
    plt.xticks([], [])
    sub1.set_xticks([])
    sub1.set_xticks([], minor=True)
    sub1.set_xticks([1, 2, 4, 8], ['2^0', '2^1', '2^2', '2^3'])
    plt.ylabel("Total Error")
    plt.xlabel("1 / The Courant Number")
    plt.title('Error vs 1/cour')
    plt.legend()
    plt.savefig('error_vs_cour.png')
    plt.close()
    
    #for the uel_vs_time graph:
    u_num_arr = []
    t_num_arr = []
    u_ana_arr = []
    t_ana_arr = []
    find_uel(u_num_arr, t_num_arr, u_ana_arr, t_ana_arr)
    fig2 = plt.figure()
    #plt.plot(t_ana_arr, u_ana_arr, 'r.')
    plt.plot(t_num_arr, u_num_arr, 'b.', label = 'numerical')
    plt.ylabel("internal energy")
    plt.xlabel("time")
    #plt.legend()
    plt.savefig('u_vs_t.png')
    plt.close()
    """
    th_want = np.pi/2
    plot_conv_res(10, th_want)
    plot_conv_cour(10, th_want)
    

