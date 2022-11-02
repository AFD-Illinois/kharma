import sys
import numpy as np
import h5py

from pyharm.grid import make_some_grid
from pyharm.defs import Loci, Slices

f = h5py.File(sys.argv[1], "r")
B = f['p'][5:8,:,:,:].transpose(0,3,2,1)

G = make_some_grid('fmks', 288, 128, 128, 0.9375)
gdet = G.gdet[Loci.CENT.value]
s = Slices(ng=1)

divB = np.abs(0.25 * (
        B[0][s.b, s.b, s.b] * gdet[s.b, s.b, :]
        + B[0][s.b, s.l1, s.b] * gdet[s.b, s.l1, :]
        + B[0][s.b, s.b, s.l1] * gdet[s.b, s.b, :]
        + B[0][s.b, s.l1, s.l1] * gdet[s.b, s.l1, :]
        - B[0][s.l1, s.b, s.b] * gdet[s.l1, s.b, :]
        - B[0][s.l1, s.l1, s.b] * gdet[s.l1, s.l1, :]
        - B[0][s.l1, s.b, s.l1] * gdet[s.l1, s.b, :]
        - B[0][s.l1, s.l1, s.l1] * gdet[s.l1, s.l1, :]
        ) / G.dx[1] + 0.25 * (
        B[1][s.b, s.b, s.b] * gdet[s.b, s.b, :]
        + B[1][s.l1, s.b, s.b] * gdet[s.l1, s.b, :]
        + B[1][s.b, s.b, s.l1] * gdet[s.b, s.b, :]
        + B[1][s.l1, s.b, s.l1] * gdet[s.l1, s.b, :]
        - B[1][s.b, s.l1, s.b] * gdet[s.b, s.l1, :]
        - B[1][s.l1, s.l1, s.b] * gdet[s.l1, s.l1, :]
        - B[1][s.b, s.l1, s.l1] * gdet[s.b, s.l1, :]
        - B[1][s.l1, s.l1, s.l1] * gdet[s.l1, s.l1, :]
        ) / G.dx[2] + 0.25 * (
        B[2][s.b, s.b, s.b] * gdet[s.b, s.b, :]
        + B[2][s.b, s.l1, s.b] * gdet[s.b, s.l1, :]
        + B[2][s.l1, s.b, s.b] * gdet[s.l1, s.b, :]
        + B[2][s.l1, s.l1, s.b] * gdet[s.l1, s.l1, :]
        - B[2][s.b, s.b, s.l1] * gdet[s.b, s.b, :]
        - B[2][s.b, s.l1, s.l1] * gdet[s.b, s.l1, :]
        - B[2][s.l1, s.b, s.l1] * gdet[s.l1, s.b, :]
        - B[2][s.l1, s.l1, s.l1] * gdet[s.l1, s.l1, :]
        ) / G.dx[3])

print("Max divB ", np.max(divB), " at ", np.unravel_index(np.argmax(divB, axis=None), divB.shape))
