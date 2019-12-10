#!/usr/bin/env python

from hdf5_to_dict import load_geom, load_hdr
import sys
import math
import numpy as np

hdr = load_hdr(sys.argv[1])
geom = load_geom(sys.argv[2])

nnodes = int(sys.argv[3])
tf = float(sys.argv[4])

SMALL = 1e-20
dt_light_min = 1./SMALL;

N1, N2, N3 = hdr['n1'], hdr['n2'], hdr['n3']

dx = [ 0, hdr['dx1'],hdr['dx2'], hdr['dx3'] ]

dt_light = np.zeros((N1,N2))

for i in range(N1):
  for j in range(N2):
    dt_light[i,j] = 1.e30
    light_phase_speed = SMALL
    dt_light_local = 0.

    for mu in range(1,4):
      if(math.pow(geom['gcon'][i,j,0,mu], 2.) -
        geom['gcon'][i,j,mu,mu]*geom['gcon'][i,j,0,0] >= 0.):
        
        cplus = np.fabs((-geom['gcon'][i,j,0,mu] +
            np.sqrt(math.pow(geom['gcon'][i,j,0,mu], 2.) -
            geom['gcon'][i,j,mu,mu]*geom['gcon'][i,j,0,0]))/
            (geom['gcon'][i,j,0,0]))

        cminus = np.fabs((-geom['gcon'][i,j,0,mu] -
            np.sqrt(math.pow(geom['gcon'][i,j,0,mu], 2.) -
            geom['gcon'][i,j,mu,mu]*geom['gcon'][i,j,0,0]))/
            (geom['gcon'][i,j,0,0]))

        light_phase_speed= max([cplus,cminus])
      else:
        light_phase_speed = SMALL

      dt_light_local += 1./(dx[mu]/light_phase_speed);

      if (dx[mu]/light_phase_speed < dt_light[i,j]):
          dt_light[i,j] = dx[mu]/light_phase_speed

    dt_light_local = 1./dt_light_local
    if (dt_light_local < dt_light_min):
      dt_light_min = dt_light_local

print("bhlight min is", dt_light_min)
#print "directional min is", np.min(dt_light)
tstep = 0.9*dt_light_min
print("timestep is then", tstep)

size = N1*N2*N3/nnodes
zcps = 813609*np.log(size) - 6327477
print("zcps per node is", zcps, ", total is", zcps*nnodes)
wall_per_step = (N1*N2*N3)/(zcps*nnodes)
print("walltime per step is", wall_per_step)
print("total time is", tf/tstep*wall_per_step/3600, " hours")
