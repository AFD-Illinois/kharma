
import numpy as np
import h5py
import matplotlib.pyplot as plt

f = h5py.File("hubble.out0.final.phdf", "r")

gam = 5/3
game = 4/3

rho0 = 1
v0 = 1e-3
ug0 = 1e-3
fel0 = 1.0
u0 = fel0 * ug0
t = 1000

x = np.linspace(0.0, 1.0, 128)
kap = (gam - 2) * (game - 1) / (game - 2) * u0 / rho0**game * (1 + v0 * t)**(game - 2)
kap_dump = f['prims.Kel_Constant'][0,0,0,:,0]

fig, ax = plt.subplots(2,2, figsize=(10,10))
ax[0, 0].plot(x,f['prims.uvec'][0,0,0,:,0])
ax[0, 0].plot(x, v0*x / (1 + v0 * t))
ax[0, 0].set_title("vx")

ax[0, 1].plot(x,f['prims.rho'][0,0,0,:,0])
ax[0, 1].plot(x, rho0 / (1 + v0 * t) * np.ones_like(x))
ax[0, 1].set_title("rho")

ax[1, 0].plot(x,f['prims.u'][0,0,0,:,0])
ax[1, 0].plot(x, ug0 / (1 + v0 * t)**2 * np.ones_like(x))
ax[1, 0].set_title("u")

kap = (gam - 2) * (game - 1) / (game - 2) * u0 / rho0**game * (1 + v0 * t)**(game - 2)
ax[1, 1].plot(x, f['prims.Kel_Constant'][0,0,0,:,0])
ax[1, 1].plot(x, kap*np.ones_like(x))
ax[1, 1].set_title("kappa_e")

plt.savefig("hubble.png")