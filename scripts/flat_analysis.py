import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprocessing as mp
import pyharm.fluid_dump as fluid_dump
import sys
sys.path.append('/home/samason4/kharma_cooling/pyharm')

# python3 ./scripts/flat_analysis.py

def find_error(dumpsdir, i, j):
    #numerical:
    dump_kharma = fluid_dump.load_dump("./dumps{0:01}/flat_space.out0.00009.phdf".format(dumpsdir))
    rho = dump_kharma['rho'][i,j,0]
    kel = dump_kharma['Kel_Howes'][i,j,0]
    t = dump_kharma['t']
    game = 1.333333
    u_num = rho**game*kel/(game-1)
    print("t with courant=", dumpsdir, ":", t)
    print("numerical u with courant=", dumpsdir, ":", u_num)
    #analytical:
    dump_kharma2 = fluid_dump.load_dump("./dumps{0:01}/flat_space.out0.00000.phdf".format(dumpsdir))
    rho2 = dump_kharma2['rho'][i,j,0]
    kel2 = dump_kharma2['Kel_Howes'][i,j,0]
    t2 = dump_kharma2['t']
    u0 = rho2**game*kel2/(game-1)
    alpha = -0.2
    u_ana = u0*np.exp(alpha*float(t))
    print("analytical u with courant=", dumpsdir, ":", u_ana)
    #print("kel2 with courant=", dumpsdir, ":", kel2)
    print("u0 with courant=", dumpsdir, ":", u0)
    #error:
    return (abs(u_num-u_ana))

def find_uel(u_num_arr, t_num_arr, u_ana_arr, t_ana_arr):
    for i in range(31):
        dump_kharma = fluid_dump.load_dump("./flat_space.out0.{0:05}.phdf".format(i))
        rho = dump_kharma['rho'][5,5,0]
        kel = dump_kharma['Kel_Howes'][5,5,0]
        #keltot = dump_kharma['Ktot'][5,5,0]
        print("kel_howes at ", i, ": ", kel)
        #print("Ktot at ", i, ": ", keltot)
        t = dump_kharma['t']
        game = 1.333333
        u_num = rho**game*kel/(game-1)
        u_num_arr.append(u_num)
        t_num_arr.append(float(t))
    dump_kharma = fluid_dump.load_dump("./flat_space.out0.00000.phdf")
    rho = dump_kharma['rho'][5,5,0]
    kel = dump_kharma['Kel_Howes'][5,5,0]
    game = 1.333333
    u_ana = rho**game*kel/(game-1)
    for i in range(31):
        dump_kharma2 = fluid_dump.load_dump("./flat_space.out0.{0:05}.phdf".format(i))
        t = dump_kharma2['t']
        u_ana_arr.append(u_ana*np.exp(-0.2*float(t)))
        t_ana_arr.append(float(t))

def find_uel5(u_num_arr, t_num_arr, u_ana_arr, t_ana_arr):
    for i in range(19):
        dump_kharma = fluid_dump.load_dump("./dumps5/flat_space.out0.{0:05}.phdf".format(i))
        rho = dump_kharma['rho'][5,5,0]
        kel = dump_kharma['Kel_Howes'][5,5,0]
        print("kel at ", i)
        t = dump_kharma['t']
        game = 1.333333
        u_num = rho**game*kel/(game-1)
        u_num_arr.append(u_num)
        t_num_arr.append(float(t))
    dump_kharma = fluid_dump.load_dump("./dumps5/flat_space.out0.00000.phdf")
    rho = dump_kharma['rho'][5,5,0]
    kel = dump_kharma['Kel_Howes'][5,5,0]
    game = 1.333333
    u_ana = rho**game*kel/(game-1)
    for i in range(19):
        dump_kharma2 = fluid_dump.load_dump("./dumps5/flat_space.out0.{0:05}.phdf".format(i))
        t = dump_kharma2['t']
        u_ana_arr.append(u_ana*np.exp(-0.2*float(t)))
        t_ana_arr.append(float(t))

def find_uel2(u_num_arr, t_num_arr, u_ana_arr, t_ana_arr):
    for i in range(19):
        dump_kharma = fluid_dump.load_dump("./dumps2/flat_space.out0.{0:05}.phdf".format(i))
        rho = dump_kharma['rho'][5,5,0]
        kel = dump_kharma['Kel_Howes'][5,5,0]
        t = dump_kharma['t']
        game = 1.333333
        u_num = rho**game*kel/(game-1)
        u_num_arr.append(u_num)
        t_num_arr.append(float(t))
    dump_kharma = fluid_dump.load_dump("./dumps2/flat_space.out0.00000.phdf")
    rho = dump_kharma['rho'][5,5,0]
    kel = dump_kharma['Kel_Howes'][5,5,0]
    game = 1.333333
    u_ana = rho**game*kel/(game-1)
    for i in range(19):
        dump_kharma2 = fluid_dump.load_dump("./dumps2/flat_space.out0.{0:05}.phdf".format(i))
        t = dump_kharma2['t']
        u_ana_arr.append(u_ana*np.exp(-0.2*float(t)))
        t_ana_arr.append(float(t))

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
    """
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
