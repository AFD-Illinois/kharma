import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprocessing as mp
import pyharm.coordinates as coords
import pyharm.fluid_dump as fluid_dump
import sys
sys.path.append('/home/samason4/kharma_cooling/pyharm')

# python3 ./scripts/bondi_analysis.py

def rho_temp_r(j, k):
    #numerical:
    dump_kharma2 = fluid_dump.load_dump("./bondi.out0.{0:05}.phdf".format(0))
    dump_kharma = fluid_dump.load_dump("./bondi.out0.final.phdf")
    th = dump_kharma['th'][:,0,0]
    print(th)
    vel1 = dump_kharma['u1'][60,j,k]
    vel2 = dump_kharma['u2'][60,j,k]
    vel3 = dump_kharma['u3'][60,j,k]
    #print("vel1: ", vel1, ", vel2: ", vel2, ", vel3: ", vel3)
    rhos = []
    temps = []
    rs = []
    errors = []
    kel2s = []
    kels = []
    for i in range (128):
        rho2 = dump_kharma2['rho'][i,j,k]
        u2 = dump_kharma2['u'][i,j,k]
        r2 = dump_kharma2['r'][i,j,k]
        kel2 = dump_kharma2['Kel_Howes'][i,j,k]
        gam2 = 1.6666666
        #not sure how to get temp but I think it's somehting like temp = (1-gam)*uion/rho or somehting like that.
        #In SI there would be mass/(boltzmann's constant) multiplied by that
        temp2 = (gam2-1)*u2/rho2
        rho = dump_kharma['rho'][i,j,k]
        u = dump_kharma['u'][i,j,k]
        r = dump_kharma['r'][i,j,k]
        kel = dump_kharma['Kel_Howes'][i,j,k]
        gam = 1.6666666
        game = 1.333333
        #not sure how to get temp but I think it's somehting like temp = (1-gam)*uion/rho or somehting like that.
        #In SI there would be mass/(boltzmann's constant) multiplied by that
        temp = (gam-1)*u/rho
        drho = rho-rho2
        dtemp = temp-temp2
        dr = r-r2
        du = u-u2
        dkel = (kel-kel2)
        #print(kel, kel2)
        rhos.append(rho)
        temps.append(temp)
        rs.append(r)
        errors.append(abs(dkel))
        kel2s.append(kel2)
        kels.append(kel)
        #print(i, "/127")
        #print("uel = ", rho**game*kel/(game-1))
    return [rhos, temps, rs, errors, kel2s, kels]

def plot_rho_temp_r():
    arr = rho_temp_r(0, 0)
    rhos = arr[0]
    temps = arr[1]
    rs = arr[2]
    errors = arr[3]
    """fig1 = plt.figure()
    plt.semilogy(rs, rhos, 'b', label = 'rho')
    plt.semilogy(rs, temps, 'r', label = 'temperature')
    plt.xlabel("radius")
    plt.title("rho and temp vs radius for bondi with no electron cooling")
    plt.legend()
    plt.savefig('rho_and_temp_vs_r2.png')"""
    fig2 = plt.figure()
    plt.plot(rs, arr[4], 'g', label = 'kel at t=0')
    plt.plot(rs, arr[5], 'r', label = 'kel at t=500')
    #plt.semilogy(rs, errors, 'b', label = 'difference between the two')
    plt.xlabel("radius")
    plt.title("kel vs r")
    plt.ylabel("kel")
    plt.legend()
    plt.savefig('Kel_at_t=500_third.png')
    plt.close()

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

def plot_conv_res(i):
    u128 = find_error128(i)[0]
    u256 = find_error256(i)[0]
    fig1, sub1 = plt.subplots()
    plt.loglog([128, 256], [u128, u256], 'b', label = 'numerical error')
    plt.loglog([128, 256], [u128+1e-13, u128*(2)**(-2.0)+1e-13], 'r', label = 'line of slope N^-2')
    plt.xticks([], [])
    sub1.set_xticks([])
    sub1.set_xticks([], minor=True)
    sub1.set_xticks([128,256])
    sub1.set_xticklabels(['128','256'])
    plt.xlabel("resolution (the simulations were nx1x1)")
    plt.ylabel("total error in internal energy")
    plt.title("error vs resolution for bondi test")
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

def find_error256(i):
    dump_kharma = fluid_dump.load_dump("./dumps256/bondi.out0.{0:05}.phdf".format(0))
    U1 = dump_kharma['u1'][i,0,0]
    U2 = dump_kharma['u2'][i,0,0]
    U3 = dump_kharma['u3'][i,0,0]
    rho = dump_kharma['rho'][i,0,0]
    gcov = dump_kharma['gcov']
    lapse = dump_kharma['lapse'][i,0,0]
    kel = dump_kharma['Kel_Howes'][i,0,0]
    game = 1.333333
    qsq1 = gcov[1,1,i,0,0]*U1*U1+gcov[2,2,i,0,0]*U2*U2+gcov[3,3,i,0,0]*U3*U3
    qsq2 = gcov[1,2,i,0,0]*U1*U2+gcov[1,3,i,0,0]*U1*U3+gcov[2,3,i,0,0]*U2*U3
    qsq = qsq1+2*qsq2
    gamma = (1+qsq)**0.5
    ut = gamma/lapse
    r = dump_kharma['r'][i,0,0]
    m = 3.0
    alpha = -1*r**(-3/2)*m/ut
    u_init = rho**game*kel/(game-1)

    dump_kharma2 = fluid_dump.load_dump("./dumps256/bondi.out0.final.phdf")
    rho2 = dump_kharma2['rho'][i,0,0]
    kel2 = dump_kharma2['Kel_Howes'][i,0,0]
    game2 = 1.333333
    t = dump_kharma2['t']
    u_num = rho2**game2*kel2/(game2-1)
    u_ana = u_init*np.exp(alpha*float(t))
    return [abs(u_num-u_ana), t]

def find_error128(i):
    dump_kharma = fluid_dump.load_dump("./dumps128/bondi.out0.{0:05}.phdf".format(0))
    U1 = dump_kharma['u1'][i,0,0]
    U2 = dump_kharma['u2'][i,0,0]
    U3 = dump_kharma['u3'][i,0,0]
    rho = dump_kharma['rho'][i,0,0]
    gcov = dump_kharma['gcov']
    lapse = dump_kharma['lapse'][i,0,0]
    kel = dump_kharma['Kel_Howes'][i,0,0]
    game = 1.333333
    qsq1 = gcov[1,1,i,0,0]*U1*U1+gcov[2,2,i,0,0]*U2*U2+gcov[3,3,i,0,0]*U3*U3
    qsq2 = gcov[1,2,i,0,0]*U1*U2+gcov[1,3,i,0,0]*U1*U3+gcov[2,3,i,0,0]*U2*U3
    qsq = qsq1+2*qsq2
    gamma = (1+qsq)**0.5
    ut = gamma/lapse
    r = dump_kharma['r'][i,0,0]
    m = 3.0
    alpha = -1*r**(-3/2)*m/ut
    u_init = rho**game*kel/(game-1)

    dump_kharma2 = fluid_dump.load_dump("./dumps128/bondi.out0.final.phdf")
    rho2 = dump_kharma2['rho'][i,0,0]
    kel2 = dump_kharma2['Kel_Howes'][i,0,0]
    game2 = 1.333333
    t = dump_kharma2['t']
    u_num = rho2**game2*kel2/(game2-1)
    u_ana = u_init*np.exp(alpha*float(t))
    return [abs(u_num-u_ana), t]

def find_error9(i):
    dump_kharma = fluid_dump.load_dump("./dumps9/bondi.out0.{0:05}.phdf".format(0))
    U1 = dump_kharma['u1'][i,0,0]
    U2 = dump_kharma['u2'][i,0,0]
    U3 = dump_kharma['u3'][i,0,0]
    rho = dump_kharma['rho'][i,0,0]
    gcov = dump_kharma['gcov']
    lapse = dump_kharma['lapse'][i,0,0]
    kel = dump_kharma['Kel_Howes'][i,0,0]
    game = 1.333333
    qsq1 = gcov[1,1,i,0,0]*U1*U1+gcov[2,2,i,0,0]*U2*U2+gcov[3,3,i,0,0]*U3*U3
    qsq2 = gcov[1,2,i,0,0]*U1*U2+gcov[1,3,i,0,0]*U1*U3+gcov[2,3,i,0,0]*U2*U3
    qsq = qsq1+2*qsq2
    gamma = (1+qsq)**0.5
    ut = gamma/lapse
    r = dump_kharma['r'][i,0,0]
    m = 3.0
    alpha = -1*r**(-3/2)*m/ut
    u_init = rho**game*kel/(game-1)

    dump_kharma2 = fluid_dump.load_dump("./dumps9/bondi.out0.final.phdf")
    rho2 = dump_kharma2['rho'][i,0,0]
    kel2 = dump_kharma2['Kel_Howes'][i,0,0]
    game2 = 1.333333
    t = dump_kharma2['t']
    u_num = rho2**game2*kel2/(game2-1)
    u_ana = u_init*np.exp(alpha*float(t))
    return [abs(u_num-u_ana), t]

def find_error5(i):
    dump_kharma = fluid_dump.load_dump("./dumps5/bondi.out0.{0:05}.phdf".format(0))
    U1 = dump_kharma['u1'][i,0,0]
    U2 = dump_kharma['u2'][i,0,0]
    U3 = dump_kharma['u3'][i,0,0]
    rho = dump_kharma['rho'][i,0,0]
    gcov = dump_kharma['gcov']
    lapse = dump_kharma['lapse'][i,0,0]
    kel = dump_kharma['Kel_Howes'][i,0,0]
    game = 1.333333
    qsq1 = gcov[1,1,i,0,0]*U1*U1+gcov[2,2,i,0,0]*U2*U2+gcov[3,3,i,0,0]*U3*U3
    qsq2 = gcov[1,2,i,0,0]*U1*U2+gcov[1,3,i,0,0]*U1*U3+gcov[2,3,i,0,0]*U2*U3
    qsq = qsq1+2*qsq2
    gamma = (1+qsq)**0.5
    ut = gamma/lapse
    r = dump_kharma['r'][i,0,0]
    m = 3.0
    alpha = -1*r**(-3/2)*m/ut
    u_init = rho**game*kel/(game-1)

    dump_kharma2 = fluid_dump.load_dump("./dumps5/bondi.out0.final.phdf")
    rho2 = dump_kharma2['rho'][i,0,0]
    kel2 = dump_kharma2['Kel_Howes'][i,0,0]
    game2 = 1.333333
    t = dump_kharma2['t']
    u_num = rho2**game2*kel2/(game2-1)
    u_ana = u_init*np.exp(alpha*float(t))
    return [abs(u_num-u_ana), t]

def plot_conv_cour(i):
    u9 = find_error128(i)[0]
    u5 = find_error5(i)[0]
    print("u9: ", u9)
    print("u5: ", u5)
    fig1, sub1 = plt.subplots()
    plt.loglog([1/0.9, 1/0.5], [u9, u5], 'b', label = 'numerical error')
    plt.loglog([1/0.9, 1/0.5], [u9, u5], 'b.')
    plt.loglog([1/0.9, 1/0.5], [u9+0.5e-12, u9*(2)**(-2.0)+0.5e-12], 'r', label = 'line of slope N^-2')
    #plt.ylim(8.3*10**-13, 8.5*10**-13)
    plt.xticks([], [])
    sub1.set_xticks([])
    sub1.set_xticks([], minor=True)
    sub1.set_xticks([1/0.9,1/0.5])
    sub1.set_xticklabels(['1/0.9','1/0.5'])
    plt.xlabel("1/(courant number)")
    plt.ylabel("total error in internal energy")
    plt.title("error vs courant for bondi test")
    plt.legend()
    plt.savefig('error_conv_cour.png')
    plt.close()

def find_ind(r_want, th_want, itot, jtot):
    dump_kharma = fluid_dump.load_dump("./bondi.out0.00000.phdf")
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

def plot_rho_change():
    rhos = []
    for i in range(128):
        dump_kharma = fluid_dump.load_dump("./dumps128/bondi.out0.{0:05}.phdf".format(0))
        rho = dump_kharma['rho'][i,0,0]
        dump_kharma2 = fluid_dump.load_dump("./dumps128/bondi.out0.final.phdf")
        rho2 = dump_kharma2['rho'][i,0,0]
        print(abs(rho2-rho))
        rhos.append(abs(rho2-rho))

    """
    fig1, sub1 = plt.subplots()
    for i in range(9):
        plt.loglog(rhos[i][1], rhos[i][0], 'b', label = 'numerical error')
    plt.xticks([], [])
    sub1.set_xticks([])
    sub1.set_xticks([], minor=True)
    sub1.set_xticks([1/0.9,1/0.5])
    sub1.set_xticklabels(['1/0.9','1/0.5'])
    plt.xlabel("1/(courant number)")
    plt.ylabel("total error in internal energy")
    plt.title("error vs courant for bondi test")
    plt.legend()
    plt.savefig('error_conv_cour.png')
    plt.close()"""

if __name__=="__main__":
    plot_rho_temp_r()
    #plot_conv_cour(60)
    #plot_rho_change()
    